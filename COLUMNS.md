# 结果 CSV 字段说明

`results/*/estimates.csv` 的每行是一次 IMU 时刻的输出，列顺序由 [columns.py](columns.py) 统一定义。初始化等待区间不输出在线估计。

| 列 / 列组 | 含义、坐标系与单位 |
|---|---|
| `timestamp` | 原始时间基准的十进制秒文本，保留纳秒；读取时勿先转为大数浮点 |
| `time_s` | 相对第一帧 IMU 的秒数 |
| `x_m,y_m,z_m` | IMU 原点在世界系中的位置，m |
| `vx_m_s,vy_m_s,vz_m_s` | 世界系速度，m/s |
| `qx,qy,qz,qw` | Hamilton 单位四元数，机体系 → 世界系；q 与 −q 等价 |
| `bgx_rad_s,bgy_rad_s,bgz_rad_s` | 机体系陀螺零偏，rad/s |
| `bax_m_s2,bay_m_s2,baz_m_s2` | 机体系加计零偏，m/s² |
| `P_px,P_py,P_pz` | 位置误差方差，m² |
| `P_vx,P_vy,P_vz` | 速度误差方差，(m/s)² |
| `P_theta_x,P_theta_y,P_theta_z` | 右局部旋转误差方差，rad²；不是欧拉角方差 |
| `P_bgx,P_bgy,P_bgz` | 陀螺零偏误差方差，(rad/s)² |
| `P_bax,P_bay,P_baz` | 加计零偏误差方差，(m/s²)² |
| `latest_nis` | 最近一次位姿更新的 NIS；观测之间会保持该值，**不代表当前输出帧发生新更新**；首次更新前为空缺值 NaN |
| `roll_rad,pitch_rad,yaw_rad` | ZYX 约定的欧拉角，rad |
| `roll_deg,pitch_deg,yaw_deg` | 同上，deg；yaw 在 ±180° 处会换支 |
| `yaw_unwrapped_deg` | 用于显示连续转动的展开 yaw，deg，不参与滤波 |

P 是模型给出的误差协方差。真实数据存在同源与时间相关性，不能直接把它作为已校准的真实置信度。CSV 保存的是 P 的 15 个对角项，不是完整的 15×15 矩阵。

## 创新和积分基线

`innovations.csv` 只在位姿观测更新时写入：`time_s` 为事件时刻；`rx_m,ry_m,rz_m` 为更新前位置残差；`rtheta_x_rad,rtheta_y_rad,rtheta_z_rad` 为 `Log(Rpredᵀ Robs)`；`nis` 为该次更新的归一化创新平方。默认回放不启用 NIS 门限。

`baselines.csv` 保存同一初始条件下的两条积分轨迹：`raw_*` 是未补偿零偏的 IMU 积分，`bias_*` 是只补偿初始零偏的积分；位置单位 m，四元数顺序 xyzw。这里 `bias_*` 是轨迹前缀，不是偏置状态列。

`summary.json` 的 `input_pose_disagreement_*` 只比较确切同刻的输入位姿。没有同刻样本时为 null；没有位姿更新时 NIS 汇总为 null。这些指标不表示真实数据的独立精度。
