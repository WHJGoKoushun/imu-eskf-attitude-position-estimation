"""回放数组的列约定；CSV 在这些状态列前加 timestamp，后加欧拉角。"""

# result["values"]：时间、名义状态、15 维误差协方差对角线、最近一次 NIS。
COL_TIME = 0
COL_P = slice(1, 4)
COL_V = slice(4, 7)
COL_Q = slice(7, 11)
COL_BG = slice(11, 14)
COL_BA = slice(14, 17)
COL_P_DIAG = slice(17, 32)
COL_LATEST_NIS = 32

ERROR_STATE_NAMES = ("px", "py", "pz", "vx", "vy", "vz", "theta_x", "theta_y", "theta_z",
                     "bgx", "bgy", "bgz", "bax", "bay", "baz")
COL_THETA_Z_VAR = COL_P_DIAG.start + ERROR_STATE_NAMES.index("theta_z")
VALUE_HEADERS = ("time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                 "qx", "qy", "qz", "qw", "bgx_rad_s", "bgy_rad_s", "bgz_rad_s",
                 "bax_m_s2", "bay_m_s2", "baz_m_s2") + tuple(
                     "P_" + name for name in ERROR_STATE_NAMES) + ("latest_nis",)
EULER_HEADERS = ("roll_rad", "pitch_rad", "yaw_rad", "roll_deg", "pitch_deg", "yaw_deg",
                 "yaw_unwrapped_deg")

# result["innovations"]：只记录实际尝试更新的事件；与每个 IMU 输出点分开。
INNOV_TIME = 0
INNOV_NIS = 7
INNOV_HEADERS = ("time_s", "rx_m", "ry_m", "rz_m", "rtheta_x_rad", "rtheta_y_rad",
                 "rtheta_z_rad", "nis")

# result["baselines"]：原始积分与初始零偏补偿积分的并排输出。
BASE_RAW_P = slice(1, 4)
BASE_RAW_Q = slice(4, 8)
BASE_CORRECTED_P = slice(8, 11)
BASE_CORRECTED_Q = slice(11, 15)
BASE_HEADERS = ("time_s", "raw_x", "raw_y", "raw_z", "raw_qx", "raw_qy", "raw_qz", "raw_qw",
                "bias_x", "bias_y", "bias_z", "bias_qx", "bias_qy", "bias_qz", "bias_qw")
