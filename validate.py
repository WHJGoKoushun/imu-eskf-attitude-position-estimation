"""Independent numerical tests and analytic-truth experiments for the ESKF.

Run: python validate.py --output results/validation

Truth comes from differentiable analytic position/Euler-angle functions, not
from the ESKF. SciPy rotations and matrix exponentials serve as independent
references. Synthetic measurement errors are independent across samples and
from IMU noise; this is deliberately different from the supplied FAST-LIO data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from eskf import (Config, ESKF, discretize, right_jacobian,
                  right_jacobian_inverse, rpy_right_jacobian)


GRAVITY = np.array([0.0, 0.0, -9.81])


def state_error(reference, other):
    """Right/local attitude error, with PDF ordering p,v,theta,bg,ba."""
    attitude = (Rotation.from_quat(reference.q).inv()
                * Rotation.from_quat(other.q)).as_rotvec()
    return np.r_[other.p-reference.p, other.v-reference.v, attitude,
                 other.bg-reference.bg, other.ba-reference.ba]


def make_filter(p=None, q=None, bg=None, ba=None, v=None, P=None, noise=None):
    return ESKF(p=np.zeros(3) if p is None else p,
                q=np.array([0., 0., 0., 1.]) if q is None else q,
                bg=np.zeros(3) if bg is None else bg,
                ba=np.zeros(3) if ba is None else ba,
                v=np.zeros(3) if v is None else v,
                P=np.eye(15)*0.01 if P is None else P,
                noise=Config(gyro_noise=0., acc_noise=0.,
                             gyro_bias_rw=0., acc_bias_rw=0.)
                if noise is None else noise,
                gravity=GRAVITY.copy())


def assert_close(actual, expected, atol, message):
    err = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    if not np.isfinite(err) or err > atol:
        raise AssertionError(f"{message}: max error {err:.6g} > {atol:.6g}")
    return err


def test_static_gravity():
    q = Rotation.from_euler("xyz", [0.3, -0.4, 0.6]).as_quat()
    bg = np.array([0.01, -0.02, 0.03])
    ba = np.array([0.08, -0.05, 0.03])
    filt = make_filter(q=q, bg=bg, ba=ba)
    force = -Rotation.from_quat(q).inv().apply(GRAVITY) + ba
    for _ in range(400):
        filt.predict(force, bg, 0.005)
    err = assert_close(filt.state.p, np.zeros(3), 2e-11,
                       "Static IMU must not accelerate in world coordinates")
    assert_close(filt.state.v, np.zeros(3), 2e-11, "Static velocity")
    return {"position_max_abs_m": err, "duration_s": 2.0,
            "initial_rpy_rad": [0.3, -0.4, 0.6]}


def test_constant_rotation_and_sign():
    q0 = Rotation.from_euler("xyz", [0.25, -0.2, 0.35]).as_quat()
    omega = np.array([0.15, -0.1, 0.45])
    one, two = make_filter(q=q0), make_filter(q=-q0)
    for _ in range(200):
        one.predict(np.array([0., 0., 9.81]), omega, 0.005)
        two.predict(np.array([0., 0., 9.81]), omega, 0.005)
    expected = Rotation.from_quat(q0) * Rotation.from_rotvec(omega)
    err = float((expected.inv()*Rotation.from_quat(one.state.q)).magnitude())
    if err > 1e-11:
        raise AssertionError(f"Body-rate rotation direction/integration: {err}")
    observation = (expected * Rotation.from_rotvec([0.015, -0.01, 0.008])).as_quat()
    cov = np.r_[np.full(3, .01**2), np.full(3, .02**2)]
    one.update_pose(np.zeros(3), observation, cov)
    two.update_pose(np.zeros(3), -observation, cov)
    sign_err = assert_close(state_error(one.state, two.state), np.zeros(15),
                            5e-10, "Quaternion double-cover equivalence")
    assert_close(one.P, two.P, 5e-10, "q and -q covariance equivalence")
    return {"attitude_error_rad": err, "sign_equivalence_max_error": sign_err}


def test_euler_covariance_mapping():
    rpy = np.array([0.45, -0.6, 0.9])
    nominal = Rotation.from_euler("xyz", rpy)
    eps = 1e-6
    jac = np.empty((3, 3))
    for j in range(3):
        d = np.eye(3)[j]*eps
        plus = (nominal.inv()*Rotation.from_euler("xyz", rpy+d)).as_rotvec()
        minus = (nominal.inv()*Rotation.from_euler("xyz", rpy-d)).as_rotvec()
        jac[:, j] = (plus-minus)/(2*eps)
    analytic = rpy_right_jacobian(nominal.as_quat())
    err = assert_close(analytic, jac, 1e-8,
                       "Euler variance must map into right attitude tangent")
    d_cov = np.diag([0.002**2, 0.003**2, 0.004**2])
    covariance_error = assert_close(analytic@d_cov@analytic.T,
                                    jac@d_cov@jac.T, 1e-12,
                                    "Euler covariance congruence")
    return {"jacobian_max_abs_error": err,
            "covariance_max_abs_error": covariance_error}


def test_discretization_against_van_loan():
    # Independent continuous kinematics in the required PDF state ordering.
    rot = Rotation.from_euler("xyz", [.35, -.4, .5]).as_matrix()
    force, omega = np.array([.4, -.3, 9.7]), np.array([.3, -.2, .8])
    def cross(v):
        x, y, z = v
        return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    F = np.zeros((15, 15))
    F[:3, 3:6] = np.eye(3)
    F[3:6, 6:9] = -rot@cross(force)
    F[3:6, 12:15] = -rot
    F[6:9, 6:9] = -cross(omega)
    F[6:9, 9:12] = -np.eye(3)
    L = np.zeros((15, 12))
    L[3:6, :3] = -rot*.03
    L[6:9, 3:6] = -np.eye(3)*.002
    L[9:12, 6:9] = np.eye(3)*.0001
    L[12:15, 9:12] = np.eye(3)*.001
    qc = L@L.T
    rows = []
    for dt in (.005, .01):
        van_loan = np.block([[F, qc], [np.zeros_like(F), -F.T]])
        exponential = expm(van_loan*dt)
        exact_phi = exponential[:15, :15]
        exact_q = exponential[:15, 15:]@exact_phi.T
        phi, qd = discretize(F, L, dt)
        # Third-order Phi is an intentional approximation, not exact expm.
        p_err = assert_close(phi, exact_phi, 2e-7, "Discrete transition")
        q_err = assert_close(qd, exact_q, 2e-10, "Integrated process noise")
        if np.linalg.eigvalsh(qd).min() < -1e-14:
            raise AssertionError("Qd must remain positive semidefinite")
        rows.append({"dt_s": dt, "phi_max_abs_error": p_err,
                     "qd_max_abs_error": q_err})
    return {"cases": rows, "reference": "scipy.linalg.expm, 30x30 Van Loan"}


def test_reset_and_residual_jacobians():
    delta = np.array([.28, -.19, .12])
    inverse_nominal = Rotation.from_rotvec(-delta)
    eps = 1e-6
    numerical = np.empty((3, 3))
    for j in range(3):
        e = np.eye(3)[j]*eps
        plus = (inverse_nominal*Rotation.from_rotvec(delta+e)).as_rotvec()
        minus = (inverse_nominal*Rotation.from_rotvec(delta-e)).as_rotvec()
        numerical[:, j] = (plus-minus)/(2*eps)
    reset_err = assert_close(right_jacobian(delta), numerical, 1e-8,
                            "Error injection reset Jacobian")
    residual = np.array([-.15, .12, .23])
    nominal = Rotation.from_rotvec(residual)
    for j in range(3):
        e = np.eye(3)[j]*eps
        plus = (nominal*Rotation.from_rotvec(e)).as_rotvec()
        minus = (nominal*Rotation.from_rotvec(-e)).as_rotvec()
        numerical[:, j] = (plus-minus)/(2*eps)
    residual_err = assert_close(right_jacobian_inverse(residual), numerical,
                               1e-8, "Log residual measurement-noise Jacobian")
    return {"reset_max_abs_error": reset_err,
            "residual_noise_max_abs_error": residual_err}


def test_predict_covariance_matches_state_derivative():
    """Check actual predict() block ordering/signs against nominal finite differences.

    The frozen continuous F is approximate for finite steps, so a short step is
    used to isolate its first-order correctness from integration truncation.
    """
    rng = np.random.default_rng(822)
    a = rng.normal(size=(15, 15))*.025
    covariance = a@a.T+np.eye(15)*.002
    p, v = np.array([.2, -.1, .3]), np.array([.4, -.2, .1])
    q = Rotation.from_euler("xyz", [.3, -.4, .7]).as_quat()
    bg, ba = np.array([.01, -.02, .005]), np.array([.1, -.03, .06])
    force, gyro = np.array([.3, -.1, 9.7]), np.array([.4, -.2, .7])
    step, epsilon = .0001, 1e-6
    nominal = make_filter(p=p, v=v, q=q, bg=bg, ba=ba, P=covariance)
    nominal.predict(force, gyro, step)
    jacobian = np.empty((15, 15))
    for axis in range(15):
        errors = []
        for sign in (1., -1.):
            d = np.eye(15)[axis]*epsilon*sign
            perturbed_q = (Rotation.from_quat(q)*Rotation.from_rotvec(d[6:9])).as_quat()
            perturbed = make_filter(p=p+d[:3], v=v+d[3:6], q=perturbed_q,
                                    bg=bg+d[9:12], ba=ba+d[12:15], P=np.zeros((15, 15)))
            perturbed.predict(force, gyro, step)
            errors.append(state_error(nominal.state, perturbed.state))
        jacobian[:, axis] = (errors[0]-errors[1])/(2*epsilon)
    reference = jacobian@covariance@jacobian.T
    error = assert_close(nominal.P, reference, 2e-9,
                         "predict covariance versus nominal state finite difference")
    return {"step_s": step, "covariance_max_abs_error": error,
            "note": "First-order model check; finite-step freeze is not exact nonlinear propagation."}


def test_update_covariance_and_zero_rejection():
    rng = np.random.default_rng(7301)
    a = rng.normal(size=(15, 15))*.025
    filt = make_filter(P=a@a.T+np.eye(15)*1e-5)
    zero_rejected = False
    try:
        filt.update_pose(np.zeros(3), [0., 0., 0., 1.], np.zeros(6))
    except ValueError:
        zero_rejected = True
    if not zero_rejected:
        raise AssertionError("Raw zero covariance must require an explicit policy")
    cov = np.r_[np.full(3, .03**2), np.full(3, .015**2)]
    for _ in range(60):
        filt.predict(np.array([0., 0., 9.81]), np.zeros(3), .005)
        filt.update_pose(rng.normal(0., .03, 3),
                         Rotation.from_rotvec(rng.normal(0., .015, 3)).as_quat(), cov)
    symmetry = assert_close(filt.P, filt.P.T, 1e-12, "Joseph/reset symmetry")
    eig = float(np.linalg.eigvalsh(filt.P).min())
    if eig < -1e-12:
        raise AssertionError(f"Posterior P is not PSD: {eig}")
    return {"zero_covariance_rejected": zero_rejected,
            "posterior_symmetry_max_abs": symmetry,
            "posterior_min_eigenvalue": eig}


def test_observation_outage():
    filt = make_filter(noise=Config(gyro_noise=.002, acc_noise=.02,
                                   gyro_bias_rw=.0001, acc_bias_rw=.001))
    before = float(np.trace(filt.P[:3, :3]))
    for _ in range(300):
        filt.predict(np.array([0., 0., 9.81]), np.zeros(3), .005)
    after = float(np.trace(filt.P[:3, :3]))
    assert_close(filt.state.p, np.zeros(3), 1e-12, "Outage propagation physics")
    if not after > before:
        raise AssertionError("Unobserved position uncertainty should grow in this case")
    return {"gap_s": 1.5, "position_variance_trace_before": before,
            "position_variance_trace_after": after}


def analytic_truth(t):
    """C2 position and orientation, static until t=2 s; explicit derivatives."""
    t = np.asarray(t, dtype=float)
    s = np.maximum(t-2., 0.)
    moving = t > 2.
    p, v, acc, euler, edot = [np.zeros(t.shape+(3,)) for _ in range(5)]
    for j, amplitude, frequency in [(0, .60, .55), (2, .12, .65)]:
        z = frequency*s
        p[..., j] = amplitude*(z-np.sin(z))
        v[..., j] = amplitude*frequency*(1.-np.cos(z))
        acc[..., j] = amplitude*frequency**2*np.sin(z)
    z, amplitude, frequency = .42*s, .60, .42
    p[..., 1] = amplitude*(1.-np.cos(z))**2
    v[..., 1] = 2*amplitude*frequency*(1.-np.cos(z))*np.sin(z)
    acc[..., 1] = 2*amplitude*frequency**2*(np.sin(z)**2+(1.-np.cos(z))*np.cos(z))
    z = .55*s
    euler[..., 0] = .12*(1.-np.cos(z))**2
    edot[..., 0] = .24*.55*(1.-np.cos(z))*np.sin(z)
    z = .40*s
    euler[..., 1] = .08*(np.sin(z)-z)
    edot[..., 1] = .08*.40*(np.cos(z)-1.)
    euler[..., 2] = .32*(z-np.sin(z))
    edot[..., 2] = .32*.40*(1.-np.cos(z))
    v *= moving[..., None]
    acc *= moving[..., None]
    edot *= moving[..., None]
    roll, pitch = euler[..., 0], euler[..., 1]
    omega = np.empty_like(euler)
    omega[..., 0] = edot[..., 0]-np.sin(pitch)*edot[..., 2]
    omega[..., 1] = np.cos(roll)*edot[..., 1]+np.sin(roll)*np.cos(pitch)*edot[..., 2]
    omega[..., 2] = -np.sin(roll)*edot[..., 1]+np.cos(roll)*np.cos(pitch)*edot[..., 2]
    rotation = Rotation.from_euler("xyz", euler)
    force = rotation.inv().apply(acc-GRAVITY)
    return {"p": p, "v": v, "a": acc, "rpy": euler, "q": rotation.as_quat(),
            "omega": omega, "force": force}


def test_analytic_truth_derivatives():
    times = np.array([0.4, 2.4, 4.5, 8.1, 15.2, 19.1])
    h = 1e-4
    center = analytic_truth(times)
    minus, plus = analytic_truth(times-h), analytic_truth(times+h)
    velocity_err = assert_close((plus["p"]-minus["p"])/(2*h), center["v"],
                                1e-8, "Independent analytic velocity derivative")
    acceleration_err = assert_close((plus["v"]-minus["v"])/(2*h), center["a"],
                                    1e-8, "Independent analytic acceleration derivative")
    r_minus, r_plus = Rotation.from_quat(minus["q"]), Rotation.from_quat(plus["q"])
    omega_numerical = (r_minus.inv()*r_plus).as_rotvec()/(2*h)
    omega_err = assert_close(omega_numerical, center["omega"], 1e-8,
                             "Independent body-rate derivative")
    return {"velocity_derivative_error": velocity_err,
            "acceleration_derivative_error": acceleration_err,
            "body_rate_derivative_error": omega_err}


def propagate_baseline(p, v, q, force, omega, dt):
    """Independent SciPy midpoint inertial integration; no ESKF code is used."""
    rotation = Rotation.from_quat(q)
    half_rotation = rotation*Rotation.from_rotvec(omega*dt*.5)
    acceleration = half_rotation.apply(force)+GRAVITY
    return (p+v*dt+.5*acceleration*dt*dt, v+acceleration*dt,
            (rotation*Rotation.from_rotvec(omega*dt)).as_quat())


def error_metrics(truth, position, quaternion):
    distance = np.linalg.norm(position-truth["p"], axis=1)
    angles = (Rotation.from_quat(truth["q"]).inv()
              * Rotation.from_quat(quaternion)).magnitude()
    return {"position_rmse_m": float(np.sqrt(np.mean(distance**2))),
            "position_max_error_m": float(distance.max()),
            "attitude_geodesic_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(angles**2)))),
            "attitude_geodesic_max_error_deg": float(np.rad2deg(angles.max()))}


def write_csv(path, names, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(np.column_stack(columns))


def simulate(seed, output, duration=20.):
    rng = np.random.default_rng(seed)
    dt, calibration_s = .005, 2.
    times = np.arange(round(duration/dt)+1)*dt
    midpoints = times[1:]-dt*.5
    truth = analytic_truth(times)
    mid_truth = analytic_truth(midpoints)
    bg = np.array([.009, -.007, .012])
    ba = np.array([.055, -.040, .070])
    gyro_sample_std, acc_sample_std = .012, .06
    position_std, euler_std = .015, .006
    gyro = mid_truth["omega"]+bg+rng.normal(0., gyro_sample_std, (len(midpoints), 3))
    force = mid_truth["force"]+ba+rng.normal(0., acc_sample_std, (len(midpoints), 3))
    n_cal = round(calibration_s/dt)
    bg0 = gyro[:n_cal].mean(axis=0)
    # The synthetic setup explicitly knows its static initial pose and gravity.
    # This is NOT an unconditional single-pose accelerometer calibration claim.
    ba0 = force[:n_cal].mean(axis=0)+GRAVITY
    initial_diag = np.r_[np.full(3, .01**2), np.full(3, .02**2),
                         np.full(3, .006**2), np.full(3, (.012/np.sqrt(n_cal))**2),
                         np.full(3, (.06/np.sqrt(n_cal))**2)]
    filt = make_filter(bg=bg0, ba=ba0, P=np.diag(initial_diag),
                       noise=Config(gyro_noise=gyro_sample_std*np.sqrt(dt),
                                    acc_noise=acc_sample_std*np.sqrt(dt),
                                    gyro_bias_rw=3e-5, acc_bias_rw=1e-4))
    count = len(times)-n_cal
    trajectories = {name: {"p": np.zeros((count, 3)), "q": np.zeros((count, 4))}
                    for name in ("raw_imu", "bias_corrected_imu", "eskf")}
    baseline_states = {name: (np.zeros(3), np.zeros(3), np.array([0., 0., 0., 1.]))
                       for name in ("raw_imu", "bias_corrected_imu")}
    for series in trajectories.values():
        series["q"][0] = [0., 0., 0., 1.]
    cov = np.r_[np.full(3, position_std**2), np.full(3, euler_std**2)]
    observation_rows, filter_times = [], []
    observed = np.zeros(count, dtype=int)
    state_trace = np.zeros((count, 10))
    state_trace[0] = np.r_[bg0, ba0, np.trace(filt.P[:3, :3]),
                           np.trace(filt.P[6:9, 6:9]), 0., 0.]
    for index in range(n_cal+1, len(times)):
        k, j, timestamp = index-1, index-n_cal, times[index]
        start = time.perf_counter()
        filt.predict(force[k], gyro[k], dt)
        # A 1.5 s observation interruption is deliberately present during motion.
        if index % 20 == 0 and not (12. <= timestamp < 13.5):
            measured_p = truth["p"][index]+rng.normal(0., position_std, 3)
            measured_euler = truth["rpy"][index]+rng.normal(0., euler_std, 3)
            measured_q = Rotation.from_euler("xyz", measured_euler).as_quat()
            outcome = filt.update_pose(measured_p, measured_q, cov,
                                        covariance_kind="rpy", gate=None)
            if outcome["status"] != "accepted":
                raise AssertionError("A valid synthetic observation was not accepted")
            observed[j] = 1
            observation_rows.append(np.r_[timestamp, measured_p, measured_q, cov])
        filter_times.append(time.perf_counter()-start)
        for name in baseline_states:
            p, v, q = baseline_states[name]
            corr_bg, corr_ba = (bg0, ba0) if name == "bias_corrected_imu" else (np.zeros(3), np.zeros(3))
            p, v, q = propagate_baseline(p, v, q, force[k]-corr_ba, gyro[k]-corr_bg, dt)
            baseline_states[name] = p, v, q
            trajectories[name]["p"][j], trajectories[name]["q"][j] = p, q
        trajectories["eskf"]["p"][j] = filt.state.p
        trajectories["eskf"]["q"][j] = filt.state.q
        state_trace[j] = np.r_[filt.state.bg, filt.state.ba,
                               np.trace(filt.P[:3, :3]), np.trace(filt.P[6:9, 6:9]),
                               np.linalg.norm(filt.state.q)-1., observed[j]]
    valid_truth = {key: value[n_cal:] for key, value in truth.items()}
    all_metrics = {name: error_metrics(valid_truth, data["p"], data["q"])
                   for name, data in trajectories.items()}
    final_eigenvalue = float(np.linalg.eigvalsh(filt.P).min())
    if final_eigenvalue < -1e-12 or not np.all(np.isfinite(filt.P)):
        raise AssertionError("Synthetic run ended with invalid covariance")
    quaternion_error = float(np.max(np.abs(state_trace[:, 8])))
    if quaternion_error > 1e-12:
        raise AssertionError("Synthetic quaternion normalization failure")
    prefix = output/f"seed_{seed}"
    headers = ["timestamp", "truth_x", "truth_y", "truth_z", "truth_qx", "truth_qy", "truth_qz", "truth_qw"]
    columns = [times[n_cal:], valid_truth["p"], valid_truth["q"]]
    for name, data in trajectories.items():
        headers += [f"{name}_{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        columns += [data["p"], data["q"]]
    headers += ["estimated_bg_x", "estimated_bg_y", "estimated_bg_z", "estimated_ba_x", "estimated_ba_y", "estimated_ba_z",
                "position_cov_trace", "rotation_cov_trace", "quaternion_norm_error", "observation_used"]
    write_csv(prefix.with_name(prefix.name+"_trajectories.csv"), headers, columns+[state_trace])
    if seed == 7:
        write_csv(output/"synthetic_imu_seed_7.csv",
                  ["interval_midpoint_timestamp", "gyro_x", "gyro_y", "gyro_z", "acc_x_mps2", "acc_y_mps2", "acc_z_mps2"],
                  [midpoints, gyro, force])
        write_csv(output/"synthetic_pose_seed_7.csv",
                  ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw",
                   "cov_00", "cov_11", "cov_22", "cov_33", "cov_44", "cov_55"],
                  [np.asarray(observation_rows)])
    return {"seed": seed, "duration_s": duration, "scored_time_interval_s": [2., duration],
            "samples_scored": count, "observations_used": len(observation_rows),
            "metrics": all_metrics, "initial_bg_estimate_radps": bg0.tolist(),
            "initial_ba_estimate_mps2": ba0.tolist(), "true_bg_radps": bg.tolist(), "true_ba_mps2": ba.tolist(),
            "final_bg_estimate_radps": filt.state.bg.tolist(), "final_ba_estimate_mps2": filt.state.ba.tolist(),
            "final_cov_min_eigenvalue": final_eigenvalue, "quaternion_max_norm_error": quaternion_error,
            "filter_step_time_ms": {"mean": float(np.mean(filter_times)*1000),
                                     "p95": float(np.percentile(filter_times, 95)*1000),
                                     "p99": float(np.percentile(filter_times, 99)*1000),
                                     "max": float(np.max(filter_times)*1000)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/validation"))
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tests = [
        (test_static_gravity, "倾斜静止状态的重力与偏置补偿", "数学与物理一致性"),
        (test_constant_rotation_and_sign, "恒角速度方向、q与−q等价性", "数学与物理一致性"),
        (test_euler_covariance_mapping, "欧拉协方差映射的有限差分", "数学与物理一致性"),
        (test_discretization_against_van_loan, "状态转移和Q对照Van Loan指数", "数值稳定性"),
        (test_reset_and_residual_jacobians, "误差重置及Log噪声雅可比", "数学与物理一致性"),
        (test_predict_covariance_matches_state_derivative, "实际预测的15维状态有限差分", "数学与物理一致性"),
        (test_update_covariance_and_zero_rejection, "Joseph后P半正定、拒绝零协方差", "数值稳定性"),
        (test_observation_outage, "无观测区间传播及P增长", "数值稳定性"),
        (test_analytic_truth_derivatives, "解析真值的位置与姿态导数", "数学与物理一致性"),
    ]
    report = {"tests": [], "simulations": [],
              "synthetic_experiment_assumptions": {
                  "truth": "Explicit analytic C2 trajectory; derivatives cross-checked numerically.",
                  "imu_rate_hz": 200, "pose_rate_hz": 10,
                  "calibration_interval_s": [0, 2], "observation_outage_s": [12, 13.5],
                  "imu_convention": "One noisy interval midpoint sample is provided at its interval end; no future pose interpolation.",
                  "initial_state": "Known static zero position/velocity and identity orientation; first 2 s used for bias initialization and excluded from scoring.",
                  "noise": "Independent sampled IMU white noise, independent Euler/position observations, constant known biases.",
                  "scope": "This controlled experiment checks filter behavior under its assumptions; it does not establish improvement over correlated real FAST-LIO measurements.",
                  "timing_scope": "Predict and observation generation/update only; excludes baseline integration, CSV I/O, and plotting."}}
    failed = False
    for test, label, category in tests:
        start = time.perf_counter()
        try:
            result = test()
            entry = {"name": test.__name__, "passed": True, "details": result}
        except Exception as error:
            failed = True
            entry = {"name": test.__name__, "passed": False, "error": str(error),
                     "traceback": traceback.format_exc()}
        entry["elapsed_s"] = time.perf_counter()-start
        entry.update(label=label, category=category)
        report["tests"].append(entry)
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {test.__name__}", flush=True)
    if not args.tests_only and not failed:
        for seed in (7, 29, 113):
            print(f"Simulating independent truth, seed {seed}...", flush=True)
            try:
                report["simulations"].append(simulate(seed, args.output))
            except Exception as error:
                report["simulations"].append({"seed": seed, "error": str(error),
                                               "traceback": traceback.format_exc()})
                failed = True
                break
        if not failed:
            metrics = {}
            for name in ("raw_imu", "bias_corrected_imu", "eskf"):
                metrics[name] = {key: float(np.mean([r["metrics"][name][key] for r in report["simulations"]]))
                                 for key in ("position_rmse_m", "attitude_geodesic_rmse_deg")}
            report["mean_metrics_across_seeds"] = metrics
    report["passed"] = not failed
    destination = args.output/"summary.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation report: {destination.resolve()}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
