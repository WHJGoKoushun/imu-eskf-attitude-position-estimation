"""临时 CSV 的端到端边界验证；--output 可保存完整 JSON 结果。"""
import argparse
import csv
import json
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_io import G, initialize, load_data, timestamp_ns, timestamp_text
from eskf import ESKF
from run import replay, save_run
from columns import COL_P


ORIGIN = 1786181234123456789
POSITION = np.array([1., -2., .4])
ROTATION = Rotation.from_euler("xyz", [.1, -.2, .3])
BG = np.array([.002, -.003, .004])
BA = np.array([.02, -.01, .03])
SETTINGS = {"initialization_seconds": 1., "gyro_bias_rw": 1e-5,
            "acc_bias_rw": 1e-4, "attitude_covariance": "rpy"}
CHECK_INFO = {
    "integer_nanosecond_precision": ("整数纳秒精度及过长小数拒绝", "时间与输入"),
    "column_aliases_and_g_conversion": ("列名别名与加速度 g 单位转换", "时间与输入"),
    "identical_duplicate_removal": ("完全相同的重复行去重", "时间与输入"),
    "conflicting_duplicate_rejection": ("冲突时间戳拒绝", "时间与输入"),
    "quaternion_sign_and_static_bias_initialization": ("四元数连续符号与已知静止偏置初始化", "初始化"),
    "sparse_initialization_rejected": ("校准窗口 IMU 不足两帧时明确拒绝", "初始化"),
    "asynchronous_event_time_and_single_consumption": ("异步观测按自身时间戳更新且只消费一次", "事件回放"),
    "zero_covariance_not_fused": ("零协方差对应的错误位姿不会被融合", "事件回放"),
    "full_observation_outage_and_csv_roundtrip": ("全程观测中断及输出 CSV 字段与物理量回读", "输出契约"),
}


def write_case(directory, aliases=False, conflict=False):
    """已知偏置的任意静止姿态；位姿错开 IMU 网格 2ms。"""
    directory.mkdir()
    keys = (["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            if aliases else ["time", "ax", "ay", "az", "gx", "gy", "gz"])
    force_g = (ROTATION.inv().apply(-G) + BA) / 9.81
    with (directory / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for i in range(401):
            row = [timestamp_text(ORIGIN + i * 5_000_000), *force_g, *BG]
            writer.writerow(row)
            if i == 50:
                repeated = row.copy()
                if conflict:
                    repeated[1] += .1
                writer.writerow(repeated)
    with (directory / "pose_cov.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
                        + [f"cov_{i}{j}" for i in range(6) for j in range(6)])
        for i in range(40):
            covariance = np.diag([.001] * 3 + [.0001] * 3)
            p = POSITION
            if i == 21:  # 1.052s：错误位姿配零协方差，必须跳过。
                covariance *= 0
                p = np.array([100., -200., 300.])
            q = ROTATION.as_quat() * (1.01 if i % 2 else -1.01)
            writer.writerow([timestamp_text(ORIGIN + 2_000_000 + i * 50_000_000),
                             *p, *q, *covariance.ravel()])


def must_reject(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("应拒绝输入，却未抛出 ValueError")


def check_pipeline(checks):
    assert timestamp_ns("1786181234.123456789") == ORIGIN
    assert timestamp_ns("1786181234.123456790") - ORIGIN == 1
    assert timestamp_text(ORIGIN) == "1786181234.123456789"
    must_reject(lambda: timestamp_ns("1786181234.1234567891"))
    checks.append("integer_nanosecond_precision")
    with TemporaryDirectory(prefix="imu_pipeline_") as temporary:
        root = Path(temporary)
        write_case(root / "canonical")
        write_case(root / "aliases", aliases=True)
        data, alias = load_data(root / "canonical"), load_data(root / "aliases")
        for key in ("imu_ns", "pose_ns", "acc", "gyro", "q"):
            np.testing.assert_array_equal(data[key], alias[key])
        assert data["origin_ns"] == ORIGIN
        np.testing.assert_array_equal(np.diff(data["imu_ns"]), 5_000_000)
        np.testing.assert_allclose(data["acc"][0], ROTATION.inv().apply(-G) + BA)
        assert data["audit"]["imu"]["identical_duplicates_removed"] == 1
        assert len(data["imu_t"]) == 401
        checks.extend(["column_aliases_and_g_conversion", "identical_duplicate_removal"])
        write_case(root / "conflict", conflict=True)
        must_reject(lambda: load_data(root / "conflict"))
        checks.append("conflicting_duplicate_rejection")
        np.testing.assert_allclose(np.linalg.norm(data["q"], axis=1), 1.)
        assert np.all(np.sum(data["q"][1:] * data["q"][:-1], axis=1) > 0)
        assert not data["cov_ok"][21]
        assert data["audit"]["zero_covariance_rows"] == 1
        initial = initialize(data, 1.)
        np.testing.assert_allclose(initial[3], BG, atol=1e-14)
        np.testing.assert_allclose(initial[4], BA, atol=1e-13)
        checks.append("quaternion_sign_and_static_bias_initialization")
        sparse = dict(data)
        for key in ("imu_t", "imu_ns", "acc", "gyro"):
            sparse[key] = data[key][[0, 200, 201]]
        must_reject(lambda: initialize(sparse, 1.))
        checks.append("sparse_initialization_rejected")

        class TracedFilter(ESKF):
            """保留真实计算，记录回放器在什么物理时刻调用更新。"""
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.elapsed, self.update_times = 0., []
                self.instances.append(self)

            def predict(self, acc, gyro, dt):
                super().predict(acc, gyro, dt)
                self.elapsed += dt

            def update_pose(self, *args, **kwargs):
                self.update_times.append(self.elapsed)
                return super().update_pose(*args, **kwargs)

        with patch("run.ESKF", TracedFilter):
            result = replay(data, SETTINGS, with_baselines=True)
        valid = (data["pose_t"] > 1.) & data["cov_ok"]
        expected_times = data["pose_t"][valid]
        np.testing.assert_allclose(TracedFilter.instances[0].update_times,
                                   expected_times - 1., atol=1e-13)
        np.testing.assert_allclose(result["innovations"][:, 0], expected_times, atol=1e-13)
        assert result["summary"]["counts"]["accepted"] == len(expected_times)
        assert result["summary"]["counts"]["invalid_covariance"] == 1
        assert len(np.unique(result["innovations"][:, 0])) == len(expected_times)
        np.testing.assert_allclose(result["values"][:, COL_P],
                                   np.tile(POSITION, (201, 1)), atol=1e-10)
        assert result["summary"]["input_pose_comparison_rows"] == 0
        assert result["summary"]["input_pose_disagreement_position_rmse_m"] is None
        checks.extend(["asynchronous_event_time_and_single_consumption", "zero_covariance_not_fused"])
        outage = replay(data, SETTINGS, dropout=[1., 3.])
        assert outage["innovations"].shape == (0, 8)
        assert outage["summary"]["nis_mean"] is None
        np.testing.assert_allclose(outage["values"][:, COL_P],
                                   np.tile(POSITION, (201, 1)), atol=1e-10)
        save_run(outage, root / "output", data["origin_ns"])
        with (root / "output" / "estimates.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 202 and all(len(row) == len(rows[0]) for row in rows)
        assert rows[1][0] == timestamp_text(ORIGIN + 1_000_000_000)
        assert rows[-1][0] == timestamp_text(ORIGIN + 2_000_000_000)
        # 独立锁定交付 CSV 的顺序与语义；不从实现常量复制期望表头。
        assert rows[0] == [
            "timestamp", "time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
            "qx", "qy", "qz", "qw", "bgx_rad_s", "bgy_rad_s", "bgz_rad_s",
            "bax_m_s2", "bay_m_s2", "baz_m_s2", "P_px", "P_py", "P_pz", "P_vx", "P_vy", "P_vz",
            "P_theta_x", "P_theta_y", "P_theta_z", "P_bgx", "P_bgy", "P_bgz", "P_bax", "P_bay", "P_baz",
            "latest_nis", "roll_rad", "pitch_rad", "yaw_rad", "roll_deg", "pitch_deg", "yaw_deg", "yaw_unwrapped_deg"]
        last_row = dict(zip(rows[0], map(float, rows[-1])))
        np.testing.assert_allclose([last_row[name] for name in ("x_m", "y_m", "z_m")], POSITION, atol=1e-10)
        np.testing.assert_allclose([last_row[name] for name in ("bgx_rad_s", "bgy_rad_s", "bgz_rad_s")], BG)
        np.testing.assert_allclose([last_row[name] for name in ("bax_m_s2", "bay_m_s2", "baz_m_s2")], BA)
        recovered_rotation = Rotation.from_quat([last_row[name] for name in ("qx", "qy", "qz", "qw")])
        assert (recovered_rotation.inv() * ROTATION).magnitude() < 1e-10
        assert np.isnan(last_row["latest_nis"])
        assert json.loads((root / "output" / "summary.json").read_text())["nis_mean"] is None
        checks.append("full_observation_outage_and_csv_roundtrip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="结果 JSON 文件路径，例如 results/validation/pipeline_checks.json")
    args = parser.parse_args()
    checks = []
    report = {"passed": True, "checks": checks}
    try:
        check_pipeline(checks)
    except Exception as error:
        report.update(passed=False, error=str(error), traceback=traceback.format_exc())
    report["check_count"] = len(checks)
    report["check_details"] = [
        {"id": name, "label": CHECK_INFO[name][0], "category": CHECK_INFO[name][1], "passed": True}
        for name in checks]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
