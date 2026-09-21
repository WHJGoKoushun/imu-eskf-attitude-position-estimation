"""题目二：按真实时间戳回放15维ESKF。python run.py --help 查看入口。"""
import os
# 小矩阵只用一个BLAS线程，避免多线程启动成本掩盖算法性能。
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
from pathlib import Path
import argparse
import csv
import json
import platform
import time
import numpy as np
from scipy.spatial.transform import Rotation
from data_io import G, load_data, initialize, timestamp_text
from eskf import ESKF, Config
from columns import (COL_TIME, COL_P, COL_Q, INNOV_NIS, INNOV_HEADERS,
                     VALUE_HEADERS, EULER_HEADERS, BASE_HEADERS)

MAX_IMU_GAP_S = 0.1
EXACT_TIME_TOLERANCE_S = 1e-8
R_SENSITIVITY_MULTIPLIERS = (10., 100.)
OBSERVATION_INTERRUPTION_S = (75., 77.)


def integrate_nominal(p, v, q, a, w, dt):
    """不做观测更新的惯导基线，输入已明确是否补偿初始零偏。"""
    rot = Rotation.from_quat(q)
    aw = (rot * Rotation.from_rotvec(w * dt / 2)).apply(a) + G
    return p + v * dt + .5 * aw * dt * dt, v + aw * dt, (
        rot * Rotation.from_rotvec(w * dt)).as_quat()


def replay(data, settings, policy="all", r_scale=1.0, with_baselines=False, dropout=None):
    start, p0, q0, bg, ba, P, initial = initialize(data, settings["initialization_seconds"])
    noise = Config(gyro_noise=initial["gyro_noise_density"],
                   acc_noise=initial["acc_noise_density"],
                   gyro_bias_rw=settings["gyro_bias_rw"], acc_bias_rw=settings["acc_bias_rw"])
    ekf = ESKF(p0, q0, bg, ba, P=P, noise=noise, gravity=G)
    ti, tp = data["imu_t"], data["pose_t"]
    out, innovations, baseline = [], [], []
    counters = {"accepted": 0, "invalid_covariance": 0, "policy_skipped": 0,
                "dropout_skipped": 0, "rejected": 0}
    j = int(np.searchsorted(tp, ti[start], side="right"))
    raw = (p0.copy(), np.zeros(3), q0.copy())
    corrected = (p0.copy(), np.zeros(3), q0.copy())
    min_eigen, max_asymmetry, max_qnorm_error = np.inf, 0., 0.
    step_durations = []
    latest_nis = float("nan")
    def record(i):
        nonlocal min_eigen, max_asymmetry, max_qnorm_error
        s = ekf.state
        qi = s.q.copy()
        if out and qi @ out[-1][COL_Q] < 0:
            qi *= -1
        out.append(np.r_[ti[i], s.p, s.v, qi, s.bg, s.ba, np.diag(ekf.P), latest_nis])
        max_asymmetry = max(max_asymmetry, float(np.max(np.abs(ekf.P - ekf.P.T))))
        max_qnorm_error = max(max_qnorm_error, float(abs(np.linalg.norm(qi) - 1)))
        # 每个输出点检查，不以抽样的最小特征值冒充全程PSD检查。
        min_eigen = min(min_eigen, float(np.linalg.eigvalsh(ekf.P)[0]))
        if with_baselines:
            baseline.append(np.r_[ti[i], raw[0], raw[2], corrected[0], corrected[2]])
    record(start)
    began = time.perf_counter()
    for i in range(start + 1, len(ti)):
        before = time.perf_counter()
        left, right = ti[i - 1], ti[i]
        elapsed = right - left
        if elapsed > MAX_IMU_GAP_S:
            raise ValueError(f"IMU {left:.6f}s 后缺口超过{MAX_IMU_GAP_S:g}秒，需另行处理，不能当连续观测。")
        cursor = left
        # 只使用当前区间端点IMU；相当于最多一个IMU周期的缓冲。
        # 观测在自己的时间戳更新，不插值未来位姿，不把一条观测重复配给多帧。
        while j < len(tp) and tp[j] <= right:
            stamp = tp[j]
            if stamp > cursor:
                alpha = ((cursor + stamp) / 2 - left) / elapsed
                a = data["acc"][i - 1] * (1 - alpha) + data["acc"][i] * alpha
                w = data["gyro"][i - 1] * (1 - alpha) + data["gyro"][i] * alpha
                ekf.predict(a, w, stamp - cursor)
                cursor = stamp
            if not data["cov_ok"][j]:
                counters["invalid_covariance"] += 1
            elif dropout is not None and dropout[0] <= stamp < dropout[1]:
                counters["dropout_skipped"] += 1
            elif policy == "cov-change" and not data["cov_changed"][j]:
                counters["policy_skipped"] += 1
            else:
                result = ekf.update_pose(data["position"][j], data["q"][j], data["diag"][j],
                                         covariance_kind=settings["attitude_covariance"], r_scale=r_scale)
                counters[result["status"]] += 1
                latest_nis = float(result["nis"])
                innovations.append(np.r_[stamp, result["residual"], latest_nis])
            j += 1
        if right > cursor:
            alpha = ((cursor + right) / 2 - left) / elapsed
            a = data["acc"][i - 1] * (1 - alpha) + data["acc"][i] * alpha
            w = data["gyro"][i - 1] * (1 - alpha) + data["gyro"][i] * alpha
            ekf.predict(a, w, right - cursor)
        if with_baselines:
            a = .5 * (data["acc"][i - 1] + data["acc"][i])
            w = .5 * (data["gyro"][i - 1] + data["gyro"][i])
            raw = integrate_nominal(*raw, a, w, elapsed)
            corrected = integrate_nominal(*corrected, a - ba, w - bg, elapsed)
        step_durations.append(time.perf_counter() - before)
        record(i)
    wall = time.perf_counter() - began
    values = np.array(out)
    # 匹配同一时刻观测仅用于“与输入观测的一致程度”，绝不当真实精度。
    ids = np.searchsorted(tp, values[:, COL_TIME], side="left")
    ids = np.minimum(ids, len(tp) - 1)
    exact = np.abs(tp[ids] - values[:, COL_TIME]) < EXACT_TIME_TOLERANCE_S
    angle = (Rotation.from_quat(values[exact, COL_Q]).inv() *
             Rotation.from_quat(data["q"][ids[exact]])).magnitude()
    perr = values[exact, COL_P] - data["position"][ids[exact]]
    innovation_array = np.array(innovations, dtype=float).reshape(-1, len(INNOV_HEADERS))
    summary = {"update_policy": policy, "R_multiplier": r_scale,
               "attitude_covariance": settings["attitude_covariance"],
               "initialization": initial, "counts": counters, "output_rows": len(values),
               "input_pose_comparison_rows": int(exact.sum()),
               "input_pose_disagreement_position_rmse_m": float(np.sqrt(np.mean(np.sum(perr**2, axis=1)))) if exact.any() else None,
               "input_pose_disagreement_attitude_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(angle**2)))) if exact.any() else None,
               "input_pose_disagreement_is_NOT_accuracy": True,
               "covariance_min_eigenvalue_all_outputs": min_eigen,
               "covariance_max_asymmetry": max_asymmetry, "quaternion_max_norm_error": max_qnorm_error,
               "wall_time_s_with_checks": wall, "processing_rate_hz_with_checks": (len(values) - 1) / wall,
               "filter_plus_baselines_step_median_ms": float(1000 * np.median(step_durations)),
               "filter_plus_baselines_step_p99_ms": float(1000 * np.quantile(step_durations, .99)),
               "filter_plus_baselines_step_max_ms": float(1000 * max(step_durations)),
               "final_bg_rad_s": ekf.state.bg.tolist(), "final_ba_m_s2": ekf.state.ba.tolist(),
               "dropout_s": dropout, "nis_mean": float(np.mean(innovation_array[:, INNOV_NIS])) if len(innovation_array) else None}
    return {"values": values, "innovations": innovation_array,
            "baselines": np.array(baseline), "summary": summary}


def save_run(result, directory, origin_ns):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    a = result["values"]
    rpy = Rotation.from_quat(a[:, COL_Q]).as_euler("xyz")
    yaw_continuous = np.unwrap(rpy[:, 2])
    head = list(VALUE_HEADERS + EULER_HEADERS)
    table = np.column_stack([a, rpy, np.rad2deg(rpy), np.rad2deg(yaw_continuous)])
    with (directory / "estimates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + head)
        for row in table:
            writer.writerow([timestamp_text(origin_ns + round(row[COL_TIME] * 1e9))] + [f"{v:.12g}" for v in row])
    np.savetxt(directory / "innovations.csv", result["innovations"], delimiter=",",
               header=",".join(INNOV_HEADERS), comments="")
    if result["baselines"].size:
        np.savetxt(directory / "baselines.csv", result["baselines"], delimiter=",", comments="",
                   header=",".join(BASE_HEADERS))
    (directory / "summary.json").write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--comparisons", action="store_true", help="增加cov-change、R敏感性及2秒观测中断实验")
    parser.add_argument("--allow-unconfirmed-frame", action="store_true", help="仅用于显式标记的探索实验，不能视作坐标系已确认")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings["pose_frame"] != "imu_to_world_confirmed" and not args.allow_unconfirmed_frame:
        parser.error("位姿参考点尚未确认。请确认pose为IMU在世界系位姿后修改config的pose_frame；探索可显式使用--allow-unconfirmed-frame。")
    data = load_data(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"settings": settings, "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "data_audit": data["audit"],
            "frame_assumption_confirmed": settings["pose_frame"] == "imu_to_world_confirmed"}
    (args.output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    experiments = [("all", "all", 1., None)]
    if args.comparisons:
        experiments.append(("cov_change", "cov-change", 1., None))
        experiments.extend((f"R_x{scale:g}", "all", scale, None)
                           for scale in R_SENSITIVITY_MULTIPLIERS)
        experiments.append(("dropout", "all", 1., OBSERVATION_INTERRUPTION_S))
    results = {}
    for name, policy, scale, gap in experiments:
        print(f"Running {name} ...", flush=True)
        result = replay(data, settings, policy=policy, r_scale=scale, with_baselines=name == "all", dropout=gap)
        result["summary"]["frame_assumption_confirmed"] = meta["frame_assumption_confirmed"]
        save_run(result, args.output / name, data["origin_ns"])
        results[name] = result
        print(json.dumps({k: result["summary"][k] for k in (
            "output_rows", "processing_rate_hz_with_checks", "covariance_min_eigenvalue_all_outputs")}), flush=True)
    from plots import real_figures, synthetic_figure
    real_figures(data, results, args.output / "figures")
    if (args.output / "validation" / "seed_7_trajectories.csv").exists():
        synthetic_figure(args.output / "validation", args.output / "figures")
    print(f"Saved results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
