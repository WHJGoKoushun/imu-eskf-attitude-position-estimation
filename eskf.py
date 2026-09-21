"""15 维右乘误差状态扩展卡尔曼滤波器（ESKF）。

标称状态：世界系位置 p、速度 v、body→world 四元数 q（xyzw）、
陀螺仪零偏 bg、加速度计零偏 ba。误差顺序 [dp,dv,dtheta,dbg,dba]，
姿态约定 R_true = R_nominal Exp(dtheta)。角速度一律使用 rad/s。
采用 Hamilton 四元数乘法；重力默认世界系 [0,0,-9.81] m/s²。
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是有限的 {length} 维向量")
    return result.copy()


def _quaternion(value):
    q = _vector(value, 4, "quaternion (xyzw)")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("四元数的模不能为零")
    return q / norm


def skew(vector):
    """[v]× w = v×w；调用者传入有限的三维向量。"""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def so3_exp(rotvec):
    """SO(3) 指数映射：弧度旋转向量 → xyzw 单位四元数。"""
    return Rotation.from_rotvec(_vector(rotvec, 3, "rotvec")).as_quat()


def so3_log(quaternion):
    """SO(3) 主值对数，长度不超过 pi；对 q 与 -q 给出相同旋转。"""
    return Rotation.from_quat(_quaternion(quaternion)).as_rotvec()


def right_jacobian(rotvec):
    """Exp(phi+dphi) ≈ Exp(phi) Exp(Jr(phi)dphi)。"""
    phi = _vector(rotvec, 3, "rotvec")
    angle2 = float(phi @ phi)
    cross = skew(phi)
    if angle2 < 1e-8:
        a = .5 - angle2 / 24 + angle2**2 / 720
        b = 1 / 6 - angle2 / 120 + angle2**2 / 5040
    else:
        angle = math.sqrt(angle2)
        a = (1 - math.cos(angle)) / angle2
        b = (angle - math.sin(angle)) / (angle2 * angle)
    return np.eye(3) - a * cross + b * (cross @ cross)


def right_jacobian_inverse(rotvec):
    """Log(Exp(phi) Exp(dphi)) ≈ phi + Jr(phi)^-1 dphi。"""
    phi = _vector(rotvec, 3, "rotvec")
    angle2 = float(phi @ phi)
    cross = skew(phi)
    if angle2 < 1e-8:
        coefficient = 1 / 12 + angle2 / 720 + angle2**2 / 30240
    else:
        angle = math.sqrt(angle2)
        coefficient = (1 - .5 * angle / math.tan(.5 * angle)) / angle2
    return np.eye(3) + .5 * cross + coefficient * (cross @ cross)


def rpy_right_jacobian(quaternion):
    """ZYX 欧拉角 [roll,pitch,yaw] 的微扰 → 右局部姿态误差。

    输入欧拉角方差必须以 rad² 表示。E 在 pitch=±pi/2 处退化，
    这是欧拉角本身的奇异性；不能把欧拉角方差当作各向同性旋转方差。
    """
    matrix = Rotation.from_quat(_quaternion(quaternion)).as_matrix()
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0]))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return np.array([[1., 0., -sp], [0., cr, sr * cp], [0., -sr, cr * cp]])


def discretize(F, noise_factor, dt):
    """冻结 F 的三阶状态转移与半正定过程噪声积分。

    noise_factor = G diag(连续白噪声幅值)，单位已含每 sqrt(s) 的缩放。
    Phi≈I+Fdt+(Fdt)²/2+(Fdt)³/6；三个 Gauss 点积分
    Qd=∫ B(s)B(s)^T ds，B(s)≈exp(Fs) noise_factor。
    这是小步长近似；大 Fdt 场景应减小步长或改用 Van Loan。
    """
    F = np.asarray(F, dtype=float)
    B0 = np.asarray(noise_factor, dtype=float)
    if (F.ndim != 2 or F.shape[0] != F.shape[1] or B0.ndim != 2
            or B0.shape[0] != F.shape[0] or not np.all(np.isfinite(F))
            or not np.all(np.isfinite(B0)) or not np.isfinite(dt) or dt < 0):
        raise ValueError("离散化输入形状、数值或 dt 无效")
    A = F * dt
    A2 = A @ A
    Phi = np.eye(F.shape[0]) + A + .5 * A2 + (A2 @ A) / 6
    B1 = F @ B0
    B2 = F @ B1
    B3 = F @ B2
    Qd = np.zeros_like(F)
    offset = math.sqrt(3 / 5)
    for node, weight in ((-offset, 5 / 9), (0., 8 / 9), (offset, 5 / 9)):
        s = .5 * dt * (node + 1)
        B = B0 + s * B1 + .5 * s**2 * B2 + s**3 / 6 * B3
        Qd += .5 * dt * weight * (B @ B.T)
    return Phi, .5 * (Qd + Qd.T)


@dataclass
class Config:
    """噪声幅值密度（不是方差，也不是单个采样的标准差）。

    gyro_noise: rad/s/sqrt(Hz)，acc_noise: m/s²/sqrt(Hz)；
    gyro_bias_rw: (rad/s)/sqrt(s)，acc_bias_rw: (m/s²)/sqrt(s)。
    默认值是可调整的工作假设，不能视为本数据传感器的已知标定值。
    """
    gyro_noise: float = .005
    acc_noise: float = .08
    gyro_bias_rw: float = .0002
    acc_bias_rw: float = .002
    max_step: float = .01

    def __post_init__(self):
        values = [self.gyro_noise, self.acc_noise,
                  self.gyro_bias_rw, self.acc_bias_rw]
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError("噪声幅值密度必须有限且非负")
        if not np.isfinite(self.max_step) or self.max_step <= 0:
            raise ValueError("max_step 必须有限且为正")


@dataclass
class State:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    bg: np.ndarray
    ba: np.ndarray


class ESKF:
    """IMU 预测，FAST-LIO 位置和姿态观测更新。

    IMU 加速度输入为比力，静止且 body 与 world 对齐时约 [0,0,9.81]。
    位姿与 IMU 必须已完成时间、单位和机体坐标系对齐。
    """

    def __init__(self, p, q, bg, ba, v=None, P=None, noise=None, gravity=None):
        self.state = State(
            _vector(p, 3, "p"), _vector(np.zeros(3) if v is None else v, 3, "v"),
            _quaternion(q), _vector(bg, 3, "bg"), _vector(ba, 3, "ba"))
        self.noise = Config() if noise is None else noise
        if not isinstance(self.noise, Config):
            raise TypeError("noise 必须为 Config")
        self.gravity = _vector([0., 0., -9.81] if gravity is None else gravity,
                               3, "gravity")
        # 默认初值标准差依次为 m、m/s、rad、rad/s、m/s²。
        if P is None:
            P = np.diag(np.repeat([1., 1., .15, .05, .5], 3)**2)
        self.P = np.asarray(P, dtype=float).copy()
        if self.P.shape != (15, 15) or not np.all(np.isfinite(self.P)):
            raise ValueError("P 必须为有限的 15×15 矩阵")
        if not np.allclose(self.P, self.P.T, atol=1e-10, rtol=1e-8):
            raise ValueError("P 必须对称")
        self.P = .5 * (self.P + self.P.T)
        if np.linalg.eigvalsh(self.P)[0] < -1e-10:
            raise ValueError("P 必须半正定")

    def predict(self, acc_mps2, gyro_radps, dt):
        """以区间中点 IMU 值预测 dt 秒；大间隔自动拆成至多 max_step。

        调用者应先将相邻 IMU 样本插值到该区间中点。本方法在每个
        子步使用中点姿态旋转比力，且在该中点冻结线性误差模型 F。
        """
        acc = _vector(acc_mps2, 3, "acc_mps2")
        gyro = _vector(gyro_radps, 3, "gyro_radps")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt 必须有限且非负")
        if dt == 0:
            return
        steps = math.ceil(dt / self.noise.max_step)
        h = dt / steps
        state = self.state
        omega, force = gyro - state.bg, acc - state.ba
        density = np.repeat([self.noise.gyro_noise, self.noise.acc_noise,
                             self.noise.gyro_bias_rw, self.noise.acc_bias_rw], 3)
        for _ in range(steps):
            rotation = Rotation.from_quat(state.q)
            midpoint = rotation * Rotation.from_rotvec(.5 * h * omega)
            R = midpoint.as_matrix()
            acceleration = R @ force + self.gravity
            state.p += state.v * h + .5 * acceleration * h**2
            state.v += acceleration * h
            state.q = (rotation * Rotation.from_rotvec(h * omega)).as_quat()
            # 右乘误差：dv_dot=-R[f]×dtheta-R dba-R na；
            # dtheta_dot=-[omega]×dtheta-dbg-ng。
            F = np.zeros((15, 15))
            F[0:3, 3:6] = np.eye(3)
            F[3:6, 6:9] = -R @ skew(force)
            F[3:6, 12:15] = -R
            F[6:9, 6:9] = -skew(omega)
            F[6:9, 9:12] = -np.eye(3)
            G = np.zeros((15, 12))
            G[6:9, 0:3] = -np.eye(3)
            G[3:6, 3:6] = -R
            G[9:12, 6:9] = np.eye(3)
            G[12:15, 9:12] = np.eye(3)
            Phi, Qd = discretize(F, G * density, h)
            self.P = Phi @ self.P @ Phi.T + Qd
            self.P = .5 * (self.P + self.P.T)

    def update_pose(self, position, quaternion, cov_diag6,
                    covariance_kind="rpy", r_scale=1., gate=None):
        """融合 [x,y,z,roll,pitch,yaw] 六个观测方差（m²、rad²）。

        covariance_kind='rpy'：把 ZYX 欧拉角方差映射至右局部姿态误差；
        'rotvec'：后三项本来就是右局部小旋转的方差。
        仅有对角数据，缺失的交叉协方差在此假设为零。非零姿态残差
        再通过 Jr^-1 映射观测噪声至 Log 坐标，是一阶 EKF 近似。
        H_theta=I 适用于小姿态创新；该变换不使大残差更新变成精确优化。
        r_scale 乘观测方差（而非标准差）；gate 为六维 NIS 阈值，
        默认不剔除观测。FAST-LIO 与 IMU 相关时，NIS 不再有精确卡方解释。
        无效协方差抛 ValueError，不替换为任意小正数。
        """
        observed_p = _vector(position, 3, "position")
        observed_q = _quaternion(quaternion)
        variances = _vector(cov_diag6, 6, "cov_diag6")
        if np.any(variances <= 0):
            raise ValueError("六个观测方差必须严格为正")
        if not np.isfinite(r_scale) or r_scale <= 0:
            raise ValueError("r_scale 必须有限且为正")
        if gate is not None and (not np.isfinite(gate) or gate <= 0):
            raise ValueError("gate 必须为正的有限 NIS 阈值或 None")
        if covariance_kind not in ("rpy", "rotvec"):
            raise ValueError("covariance_kind 只能是 'rpy' 或 'rotvec'")
        state = self.state
        nominal_rotation = Rotation.from_quat(state.q)
        delta = (nominal_rotation.inv() * Rotation.from_quat(observed_q)).as_rotvec()
        residual = np.r_[observed_p - state.p, delta]
        mapping = np.eye(6)
        angular_mapping = (rpy_right_jacobian(observed_q)
                           if covariance_kind == "rpy" else np.eye(3))
        mapping[3:6, 3:6] = right_jacobian_inverse(delta) @ angular_mapping
        measurement_cov = (mapping * (variances * r_scale)) @ mapping.T
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        PHt = self.P @ H.T
        S = H @ PHt + measurement_cov
        S = .5 * (S + S.T)
        nis = float(residual @ np.linalg.solve(S, residual))
        result = {"status": "accepted", "nis": nis, "residual": residual,
                  "correction_norm": 0., "innovation_cov": S}
        if gate is not None and nis > gate:
            result["status"] = "rejected"
            return result
        gain = np.linalg.solve(S, PHt.T).T  # 解线性方程，不显式求逆。
        correction = gain @ residual
        complement = np.eye(15) - gain @ H
        # Joseph 形式减少舍入导致的非对称或负方差。
        posterior = (complement @ self.P @ complement.T
                     + gain @ measurement_cov @ gain.T)
        state.p += correction[0:3]
        state.v += correction[3:6]
        state.q = (nominal_rotation * Rotation.from_rotvec(correction[6:9])).as_quat()
        state.bg += correction[9:12]
        state.ba += correction[12:15]
        # 注入后误差重新以新标称姿态为原点，须同步重置协方差。
        reset = np.eye(15)
        reset[6:9, 6:9] = right_jacobian(correction[6:9])
        self.P = reset @ posterior @ reset.T
        self.P = .5 * (self.P + self.P.T)
        result["correction_norm"] = float(np.linalg.norm(correction))
        return result
