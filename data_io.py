"""读取、审计和初始化。原始文件只读，时间戳先保留整数纳秒再取相对时间。"""
from pathlib import Path
import csv
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

G = np.array([0.0, 0.0, -9.81])


def timestamp_ns(value):
    """本数据为十进制秒；避免先用float读取十位Unix秒带来的精度损失。"""
    whole, _, fraction = value.strip().partition(".")
    if len(fraction) > 9 and any(c != "0" for c in fraction[9:]):
        raise ValueError("时间戳精度超过纳秒；请明确单位后转换。")
    return int(whole) * 10**9 + int((fraction + "0" * 9)[:9])


def timestamp_text(value):
    return f"{int(value) // 10**9}.{int(value) % 10**9:09d}"


def _read(path, groups, allow_bad_tail=0):
    times, rows = [], []
    dropped, duplicates = 0, 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        time_key = next((k for k in ("time", "timestamp") if k in names), None)
        keys = [next((k for k in options if k in names), None) for options in groups]
        if time_key is None or None in keys:
            raise ValueError(f"{path}: 缺少必要列，实际列为 {names}")
        for line, row in enumerate(reader, 2):
            try:
                t = timestamp_ns(row[time_key])
                values = [float(row[k]) for k in keys]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{line}: 数值或时间戳无法解析") from exc
            required = values[:-allow_bad_tail] if allow_bad_tail else values
            if not np.isfinite(required).all():
                dropped += 1
                continue
            if times and t <= times[-1]:
                if t == times[-1] and np.array_equal(values, rows[-1], equal_nan=True):
                    duplicates += 1
                    continue
                raise ValueError(f"{path}:{line}: 时间倒序或同一时间存在冲突数据，不能静默覆盖。")
            times.append(t)
            rows.append(values)
    if len(times) < 2:
        raise ValueError(f"{path}: 有效记录不足两条")
    return np.array(times, np.int64), np.array(rows), {
        "dropped_nonfinite_rows": dropped, "identical_duplicates_removed": duplicates,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def load_data(directory):
    directory = Path(directory)
    it, imu, imeta = _read(directory / "imu.csv", [
        ("ax", "acc_x"), ("ay", "acc_y"), ("az", "acc_z"),
        ("gx", "gyro_x"), ("gy", "gyro_y"), ("gz", "gyro_z")])
    pt, pose, pmeta = _read(directory / "pose_cov.csv", [
        (k,) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")
    ] + [(f"cov_{i}{j}",) for i in range(6) for j in range(6)], allow_bad_tail=36)
    norms = np.linalg.norm(pose[:, 3:7], axis=1)
    good = norms > 1e-8
    pmeta["zero_quaternions_removed"] = int((~good).sum())
    pt, pose, norms = pt[good], pose[good], norms[good]
    if len(pt) < 2:
        raise ValueError("有效位姿不足两条")
    q = pose[:, 3:7] / norms[:, None]
    flips = int((np.sum(q[1:] * q[:-1], axis=1) < 0).sum())
    # q 与 -q 表示同一旋转；统一符号只为连续输出，不改变物理姿态。
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] *= -1
    cov = pose[:, 7:].reshape(-1, 6, 6)
    diagonal = np.diagonal(cov, axis1=1, axis2=2).copy()
    cov_ok = np.isfinite(cov).all(axis=(1, 2)) & (diagonal > 0).all(axis=1)
    zero_cov = np.all(cov == 0, axis=(1, 2))
    change = np.r_[True, np.any(cov[1:] != cov[:-1], axis=(1, 2))]
    block_starts = np.flatnonzero(change)
    block_lengths = np.diff(np.r_[block_starts, len(pt)])
    lengths, counts = np.unique(block_lengths, return_counts=True)
    change_dt = np.diff(pt[block_starts]) / 1e9
    change_interval = {"count": len(change_dt)}
    for name, reducer in (("mean", np.mean), ("min", np.min),
                          ("median", np.median), ("max", np.max)):
        change_interval[name] = float(reducer(change_dt)) if len(change_dt) else None
    nonzero_cov_rows = np.flatnonzero(~zero_cov)
    origin = int(it[0])
    if min(it[-1], pt[-1]) <= max(it[0], pt[0]):
        raise ValueError("IMU 与位姿时间没有重叠。")
    def timing(t):
        dt = np.diff(t) / 1e9
        return {"rows": len(t), "duration_s": float((t[-1] - t[0]) / 1e9),
                "mean_rate_hz": float(1 / dt.mean()), "dt_min_s": float(dt.min()),
                "dt_median_s": float(np.median(dt)), "dt_max_s": float(dt.max()),
                "gaps_over_20ms": int((dt > .02).sum())}
    audit = {"imu": {**imeta, **timing(it)}, "pose": {**pmeta, **timing(pt)},
             "origin_unix_s": timestamp_text(origin),
             "exact_shared_timestamps": int(np.intersect1d(it, pt).size),
             "invalid_covariance_rows": int((~cov_ok).sum()),
             "zero_covariance_rows": int(zero_cov.sum()),
             "leading_zero_covariance_rows": int(nonzero_cov_rows[0]) if len(nonzero_cov_rows) else len(pt),
             "covariance_blocks": len(block_starts),
             "covariance_block_unique_lengths": lengths.tolist(),
             "covariance_block_length_distribution": {
                 str(length): int(count) for length, count in zip(lengths, counts)},
             "covariance_block_min_length": int(block_lengths.min()),
             "covariance_block_max_length": int(block_lengths.max()),
             "covariance_change_interval_s": change_interval,
             "quaternion_sign_flips": flips,
             "quaternion_norm_max_deviation": float(np.max(np.abs(norms - 1))),
             "acc_unit": "g in source; multiplied by 9.81 once", "gyro_unit": "rad/s",
             "timestamp_unit": "seconds; parsed as integer nanoseconds",
             "source_is_not_ground_truth": True}
    return {"imu_ns": it, "pose_ns": pt, "origin_ns": origin,
            "imu_t": (it - origin) / 1e9, "pose_t": (pt - origin) / 1e9,
            "acc": imu[:, :3] * 9.81, "gyro": imu[:, 3:],
            "position": pose[:, :3], "q": q, "cov": cov, "diag": diagonal,
            "cov_ok": cov_ok, "cov_changed": change, "audit": audit}


def initialize(data, seconds=5.0):
    """等待初始静止窗口结束后启动；不把使用了该窗口的数据伪装成零延迟输出。"""
    if seconds < .5:
        raise ValueError("初始化窗口至少0.5秒")
    ti, tp = data["imu_t"], data["pose_t"]
    i = int(np.searchsorted(ti, seconds, side="left"))
    if i >= len(ti) - 1:
        raise ValueError("数据时长不足以完成初始化并运行")
    stop = ti[i]
    # 丢开最初1秒的LIO启动瞬态；所有样本均已在启动时刻到达。
    begin = min(1.0, seconds / 3)
    mi = (ti >= begin) & (ti <= stop)
    mp = (tp >= begin) & (tp <= stop) & data["cov_ok"]
    if mi.sum() < 2:
        raise ValueError("静止窗口内有效IMU不足两帧，不能估计采样方差和噪声密度")
    if mp.sum() < 2:
        raise ValueError("静止窗口内没有足够的有效位姿协方差，不能初始化")
    a, w = data["acc"][mi], data["gyro"][mi]
    q0 = Rotation.from_quat(data["q"][mp]).mean().as_quat()
    r0 = Rotation.from_quat(q0)
    p0 = np.mean(data["position"][mp], axis=0)
    bg = w.mean(axis=0)
    ba = a.mean(axis=0) + r0.inv().apply(G)
    angle_span = np.max((r0.inv() * Rotation.from_quat(data["q"][mp])).magnitude())
    position_span = np.max(np.linalg.norm(data["position"][mp] - p0, axis=1))
    if np.max(w.std(axis=0, ddof=1)) > .03 or np.max(a.std(axis=0, ddof=1)) > .5:
        raise ValueError("初始化窗口IMU波动过大，未满足静止条件；请缩短或核实初始静止段")
    if angle_span > np.deg2rad(2) or position_span > .08:
        raise ValueError("初始化窗口位姿变化过大，不能使用静止零偏估计")
    dt = float(np.median(np.diff(ti[mi])))
    # 连续白噪声幅度：sample_std * sqrt(dt)，并非直接将样本方差乘进Q。
    ng = float(np.sqrt(np.mean(w.var(axis=0, ddof=1)) * dt))
    na = float(np.sqrt(np.mean(a.var(axis=0, ddof=1)) * dt))
    pdiag = np.r_[np.full(3, .02**2), np.full(3, .05**2),
                  np.full(3, np.deg2rad(.5)**2), np.full(3, .001**2), np.full(3, .1**2)]
    details = {"start_index": i, "start_time_s": float(stop),
               "calibration_begin_s": begin, "imu_calibration_rows": int(mi.sum()),
               "pose_calibration_rows": int(mp.sum()), "p0_m": p0.tolist(),
               "q0_xyzw": q0.tolist(), "bg0_rad_s": bg.tolist(), "ba0_m_s2": ba.tolist(),
               "gyro_sample_std_rad_s": w.std(axis=0, ddof=1).tolist(),
               "acc_sample_std_m_s2": a.std(axis=0, ddof=1).tolist(),
               "gyro_noise_density": ng, "acc_noise_density": na,
               "max_attitude_deviation_deg": float(np.rad2deg(angle_span)),
               "max_position_deviation_m": float(position_span),
               "initial_covariance_diagonal": pdiag.tolist(),
               "ba_is_conditional_on_pose_and_gravity": True}
    return i, p0, q0, bg, ba, np.diag(pdiag), details
