"""由真实数值生成图表；PNG 300 dpi 和矢量PDF同步导出。"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imu_eskf_mpl"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from columns import (COL_TIME, COL_P, COL_Q, COL_BG, COL_BA, COL_THETA_Z_VAR,
                     INNOV_TIME, INNOV_NIS, BASE_RAW_P, BASE_RAW_Q,
                     BASE_CORRECTED_P, BASE_CORRECTED_Q)

BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#777777"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .2, "axes.axisbelow": True,
                     "legend.frameon": False, "lines.linewidth": 1.3,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def euler(q):
    angles = Rotation.from_quat(q).as_euler("xyz")
    angles[:, 2] = np.unwrap(angles[:, 2])
    return np.rad2deg(angles)


def real_figures(data, results, directory):
    main = results["all"]
    values = main["values"]
    t, ref_t = values[:, COL_TIME], data["pose_t"]
    angle, ref_angle = euler(values[:, COL_Q]), euler(data["q"])
    # 相同分支原点；unwrapping仅用于显示，滤波始终在四元数上进行。
    r0 = np.searchsorted(ref_t, t[0])
    angle[:, 2] += 360 * round((ref_angle[r0, 2] - angle[0, 2]) / 360)
    for name, observed, estimated, labels in [
        ("attitude", ref_angle, angle, ["Roll (deg)", "Pitch (deg)", "Yaw, unwrapped (deg)"]),
        ("position", data["position"], values[:, COL_P], ["x (m)", "y (m)", "z (m)"])
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True, layout="constrained")
        for k, ax in enumerate(axes):
            ax.plot(ref_t, observed[:, k], color=GRAY, lw=.9, ls="--", label="FAST-LIO input observation")
            ax.plot(t, estimated[:, k], color=BLUE, label="15-state ESKF")
            ax.axvspan(0, t[0], color=ORANGE, alpha=.08)
            ax.set_ylabel(labels[k])
        axes[0].legend(loc="upper left", ncols=1)
        axes[0].set_title("Real data: " + name + " (input observations are not ground truth)", fontsize=11)
        axes[-1].set_xlabel("Time from first IMU sample (s)")
        axes[-1].set_xlim(0, data["imu_t"][-1])
        save(fig, directory, name)
    fig = plt.figure(figsize=(10, 5.5), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    trajectory = fig.add_subplot(grid[:, 0])
    position_difference = fig.add_subplot(grid[0, 1])
    attitude_difference = fig.add_subplot(grid[1, 1], sharex=position_difference)
    axes = (trajectory, position_difference, attitude_difference)
    p = values[:, COL_P]
    axes[0].plot(data["position"][:, 0], data["position"][:, 1], ls="--", color=GRAY, label="FAST-LIO input")
    axes[0].plot(p[:, 0], p[:, 1], color=BLUE, label="ESKF")
    axes[0].scatter(p[0, 0], p[0, 1], facecolors="none", edgecolors=GREEN, s=100,
                    linewidths=1.6, marker="o", label="Start after calibration", zorder=4)
    axes[0].scatter(p[-1, 0], p[-1, 1], color=ORANGE, s=35, marker="x", label="End", zorder=5)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Planar trajectory")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8)
    ids = np.minimum(np.searchsorted(ref_t, t), len(ref_t) - 1)
    mask = abs(ref_t[ids] - t) < 1e-8
    posdiff = np.linalg.norm(p[mask] - data["position"][ids[mask]], axis=1)
    adiff = np.rad2deg((Rotation.from_quat(values[mask, COL_Q]).inv() *
                       Rotation.from_quat(data["q"][ids[mask]])).magnitude())
    axes[1].plot(t[mask], posdiff * 1000, color=BLUE)
    axes[1].set(ylabel="Position difference (mm)", title="Difference from input (not accuracy)")
    axes[1].tick_params(labelbottom=False)
    axes[2].plot(t[mask], adiff, color=ORANGE, lw=.7)
    axes[2].set(xlabel="Time (s)", ylabel="Attitude difference (deg)")
    save(fig, directory, "trajectory_and_input_difference")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.7), sharex=True, layout="constrained")
    for k, (color, label) in enumerate(zip([BLUE, ORANGE, GREEN], "xyz")):
        axes[0].plot(t, values[:, COL_BG][:, k], color=color, label=label)
        axes[1].plot(t, values[:, COL_BA][:, k], color=color, label=label)
    axes[0].set(ylabel="Body gyro bias (rad/s)", title="Effective bias states; not independent sensor calibration")
    axes[1].set(ylabel="Body accelerometer bias (m/s²)", xlabel="Time (s)")
    axes[0].legend(ncols=3)
    save(fig, directory, "biases")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), layout="constrained", sharex=True)
    for name, color in [("all", BLUE), ("cov_change", ORANGE)]:
        if name in results:
            r = results[name]
            inv = r["innovations"]
            axes[0].semilogy(inv[:, INNOV_TIME], np.maximum(inv[:, INNOV_NIS], 1e-9), color=color, alpha=.7, lw=.65, label=name)
            a = r["values"]
            axes[1].plot(a[:, COL_TIME], np.rad2deg(np.sqrt(a[:, COL_THETA_Z_VAR])), color=color, label=name)
    axes[0].axhline(chi2.ppf(.95, 6), color=GRAY, ls="--", label="chi-square 95% (diagnostic only)")
    axes[0].set(ylabel="NIS (log scale)", title="Correlation violates the independent-observation confidence interpretation")
    axes[1].set(ylabel="Local theta-z model sigma (deg)", xlabel="Time (s)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save(fig, directory, "innovation_and_model_uncertainty")
    if main["baselines"].size:
        b = main["baselines"]
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained", sharex=True)
        for colp, colq, color, label in [(BASE_RAW_P, BASE_RAW_Q, ORANGE, "Raw IMU integration"),
                                       (BASE_CORRECTED_P, BASE_CORRECTED_Q, GREEN, "Initial-bias-corrected integration")]:
            pd = np.linalg.norm(b[mask, colp] - data["position"][ids[mask]], axis=1)
            ad = np.rad2deg((Rotation.from_quat(b[mask, colq]).inv() *
                            Rotation.from_quat(data["q"][ids[mask]])).magnitude())
            axes[0].semilogy(t[mask], np.maximum(pd, 1e-6), color=color, label=label)
            axes[1].plot(t[mask], ad, color=color, label=label)
        axes[0].semilogy(t[mask], np.maximum(posdiff, 1e-6), color=BLUE, label="ESKF")
        axes[1].plot(t[mask], adiff, color=BLUE, label="ESKF")
        axes[0].set(ylabel="Position difference (m, log)", title="Dead-reckoning comparison against FAST-LIO input; not ground-truth error")
        axes[1].set(ylabel="Geodesic attitude difference (deg)", xlabel="Time (s)")
        axes[0].legend(fontsize=8)
        save(fig, directory, "integration_baselines")
    if "dropout" in results:
        gap_start, gap_end = results["dropout"]["summary"]["dropout_s"]
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, layout="constrained")
        for k, ax in enumerate(axes):
            ax.plot(ref_t, data["position"][:, k], color=GRAY, ls="--", label="Input observations")
            ax.plot(t, p[:, k], color=BLUE, label="All updates")
            a = results["dropout"]["values"]
            ax.plot(a[:, COL_TIME], a[:, COL_P][:, k], color=ORANGE,
                    label=f"{gap_end - gap_start:g} s observation interruption")
            ax.axvspan(gap_start, gap_end, color=ORANGE, alpha=.1)
            ax.set_ylabel(f"{'xyz'[k]} (m)")
            ax.set_xlim(gap_start - 2, gap_end + 3)
        axes[0].set_title("Observation interruption: propagation and reacquisition")
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Time (s)")
        save(fig, directory, "observation_interruption")
    if len(results) > 1:
        labels = [k for k in results if k != "dropout"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        for ax, key, unit in zip(axes,
            ["input_pose_disagreement_position_rmse_m", "input_pose_disagreement_attitude_rmse_deg"],
            ["Position disagreement RMSE (m)", "Attitude disagreement RMSE (deg)"]):
            v = [results[k]["summary"][key] for k in labels]
            ax.bar(labels, v, color=[BLUE, ORANGE, GREEN, "#CC79A7"][:len(labels)])
            ax.set_ylabel(unit)
            ax.tick_params(axis="x", rotation=15)
        fig.suptitle("R / update-frequency sensitivity: smaller disagreement is not proof of better accuracy", fontsize=10)
        save(fig, directory, "sensitivity")


def synthetic_figure(validation_directory, figure_directory):
    table = np.genfromtxt(Path(validation_directory) / "seed_7_trajectories.csv", delimiter=",", names=True)
    t = table["timestamp"]
    def extract(prefix, names):
        return np.column_stack([table[prefix + "_" + k] for k in names])
    truth_p, truth_q = extract("truth", "xyz"), extract("truth", ["qx", "qy", "qz", "qw"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.3), layout="constrained")
    for prefix, color, label in [("raw_imu", ORANGE, "Raw IMU"),
                                 ("bias_corrected_imu", GREEN, "Bias-corrected IMU"),
                                 ("eskf", BLUE, "ESKF")]:
        p = extract(prefix, "xyz")
        q = extract(prefix, ["qx", "qy", "qz", "qw"])
        pe = np.linalg.norm(p - truth_p, axis=1)
        ae = np.rad2deg((Rotation.from_quat(truth_q).inv() * Rotation.from_quat(q)).magnitude())
        axes[0, 0].semilogy(t, np.maximum(pe, 1e-5), color=color, label=label)
        axes[0, 1].plot(t, ae, color=color, label=label)
        if prefix == "eskf":
            axes[1, 0].plot(t, pe, color=color, label="ESKF position error")
            axes[1, 1].plot(p[:, 0], p[:, 1], color=color, label="ESKF")
    axes[1, 1].plot(truth_p[:, 0], truth_p[:, 1], ls="--", color=GRAY, label="Analytic ground truth")
    axes[0, 0].set(ylabel="Position error (m, log)", title="Independent synthetic truth, seed 7")
    axes[0, 1].set(ylabel="Geodesic attitude error (deg)", title="Attitude error against known truth")
    axes[1, 0].set(ylabel="ESKF position error (m)", title="Observation interruption shaded")
    axes[1, 1].set(xlabel="x (m)", ylabel="y (m)", title="Trajectory with independent truth")
    axes[1, 1].axis("equal")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Time (s)")
        ax.axvspan(12, 13.5, color=ORANGE, alpha=.1)
        ax.legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    save(fig, figure_directory, "synthetic_validation")
