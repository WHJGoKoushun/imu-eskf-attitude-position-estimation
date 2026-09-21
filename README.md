# 题目二：IMU 姿态与位置估计

**姿态必做和位置选做均已完成。直接打开 [REPORT.pdf](REPORT.pdf) 查看 6 页验收报告，无须先运行程序。**

| 查看内容 | 入口 |
|---|---|
| 必做：Roll、Pitch、Yaw 时间曲线 | [attitude.png](results/figures/attitude.png) / [矢量 PDF](results/figures/attitude.pdf) |
| 选做：x、y、z 时间曲线 | [position.png](results/figures/position.png) / [矢量 PDF](results/figures/position.pdf) |
| 全部逐帧结果 | [estimates.csv](results/all/estimates.csv)，字段见 [COLUMNS.md](COLUMNS.md) |
| 完整方法、验证和辅助实验 | [TECHNICAL_APPENDIX.pdf](TECHNICAL_APPENDIX.pdf)；可编辑技术原稿 [REPORT.md](REPORT.md) |

算法是 **15 维右乘误差状态 EKF**：标称状态为 `p, v, q, bg, ba`，误差状态为 `[δp, δv, δθ, δbg, δba]`。15 维实现完整包含必做的姿态及陀螺零偏状态，并加入位置选做所需状态。姿态 6 维误差模型的数学形式见技术附录；运行程序统一采用 15 维联合估计。

## 核心代码地图

| 功能 | 实现 |
|---|---|
| 读取、时间解析、数据审计 | [data_io.py](data_io.py) → `load_data` |
| 初始静止零偏估计 | [data_io.py](data_io.py) → `initialize` |
| IMU 状态与协方差预测 | [eskf.py](eskf.py) → `ESKF.predict` |
| 状态转移 Φ 和过程协方差 Qd | [eskf.py](eskf.py) → `discretize` |
| 位姿残差、增益、Joseph 更新、误差重置 | [eskf.py](eskf.py) → `ESKF.update_pose` |
| 按观测时间戳回放 | [run.py](run.py) → `replay` |
| 六条时间曲线与辅助图 | [plots.py](plots.py) → `real_figures` |
| 输出列与表头的统一定义 | [columns.py](columns.py) |

## 一键复现

**Windows：** 安装 Python 3.12 后双击 `run_windows.cmd`。脚本在项目目录创建 `.venv`，安装锁定依赖，然后执行数值验证、流程检查、5 组真实数据实验和两份 PDF 生成。项目环境已有依赖时可以复用；首次安装需要网络。

**Linux：** Python 3.12 和 venv 可用后运行 `bash run_linux.sh`。本交付在 Windows 实测，Linux 脚本尚未在 Linux 系统实跑。

Python 3.12 与锁定依赖是已验证组合。当前 NumPy、SciPy 要求 Python ≥3.12，脚本会拒绝更早版本；其他满足 `3.12 ≤ 版本 < 4` 的解释器会警告后尝试安装与验证，不视为已测试兼容。数据来自现成 CSV，因此无需安装 ROS 或运行 FAST-LIO。

已有 Python 环境时也可分别执行：

```bash
python -m pip install -r requirements.txt
python validate.py --output results/validation
python validate_pipeline.py --output results/validation/pipeline_checks.json
python run.py --comparisons
python build_report.py
```

只运行主实验用 `python run.py`；数值检查用 `python validate.py --tests-only --output /path/to/test-output`，单独输出以免覆盖完整仿真报告。其他数据可通过 `--data-dir /path/to/data --output /path/to/results` 回放，两份文件仍须命名为 `imu.csv`、`pose_cov.csv`。中文技术原稿中的固定数据分析仅适用于题给记录；换数据后需重新审查这些结论。

## 数据约定

| 项目 | 约定 |
|---|---|
| 四元数 | Hamilton，`qx,qy,qz,qw`，机体系 → 世界系 |
| 欧拉角 | `R = Rz(yaw) Ry(pitch) Rx(roll)`；SciPy 外禀 `xyz` |
| 原始单位 | 加速度 g，读取时乘一次 9.81；角速度 rad/s |
| 世界重力 | `[0,0,-9.81] m/s²` |
| 时间戳 | 十进制秒，先解析成整数纳秒；使用实际时间差 |
| 观测协方差 | 按题面取六个对角项；后三项按 RPY 方差映射到旋转残差 |

输入位姿采用 IMU 在世界系中的位置与方向约定。默认收集约 5 秒初始静止数据后开始输出；加计零偏是给定初始姿态和重力下的条件估计。每条位姿在自身时间戳只使用一次，IMU 中点积分需要最多一个采样区间缓冲。异常数据策略和静止条件详见技术附录。

## 结果目录

| 目录 | 用途 |
|---|---|
| `results/all/` | 按题逐帧融合的主结果，R 倍数 1 |
| `results/cov_change/` | 仅在协方差变化时更新的对照 |
| `results/R_x10/`、`R_x100/` | 观测方差敏感性 |
| `results/dropout/` | 位姿观测短时中断 |
| `results/validation/` | 数值及流程检查、固定种子合成真值实验 |
| `results/audit/` | 题给记录的详细数据审计 |
| `results/figures/` | 必需姿态图、选做位置图，其余为附加分析 |

`results/manifest.json` 保存输入 SHA-256、参数、环境及紧凑审计；`results/delivery_checks.json` 保存本次交付验证状态。原始 CSV 副本和完整对照结果均保留在包内。

**结果解释：** 真实记录没有独立真值，FAST-LIO 与本滤波器共享 IMU。与其更贴合不证明更准确；`cov_change` 也不保证统计独立或更稳健。仿真误差只适用于所列仿真假设。相关推导、性能计时范围和数据依据集中在技术附录。
