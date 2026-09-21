"""由实验JSON生成6页验收报告和完整技术附录；REPORT.md为题给记录的技术原稿。"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imu_eskf_mpl"))
import json
import re
from xml.sax.saxutils import escape
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from matplotlib.font_manager import findfont

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results"


def real_diagnostics():
    """从已保存的主实验CSV计算诊断，不将输入差异冒充真值误差。"""
    innovation = np.genfromtxt(RESULT / "all/innovations.csv", delimiter=",", names=True)
    nis = np.atleast_1d(innovation["nis"])
    times = np.atleast_1d(innovation["time_s"])
    window = nis[(times >= 65.) & (times < 105.)]
    def stats(values):
        return {"count": len(values),
                "mean": float(np.mean(values)) if len(values) else None,
                "p99": float(np.quantile(values, .99)) if len(values) else None,
                "max": float(np.max(values)) if len(values) else None}
    answer = {"nis_all": stats(nis), "nis_window_65_105_s": stats(window),
              "nominal_dimension": 6,
              "nominal_mean_requires_correct_zero_mean_innovation_covariance": True,
              "source": "all/innovations.csv; pre-update residual, not posterior truth error",
              "causes_identified": False}
    (RESULT / "all/diagnostics.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")
    return answer


def make_report():
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    summaries = {name: read(RESULT / name / "summary.json") for name in
                 ("all", "cov_change", "R_x10", "R_x100", "dropout") if (RESULT/name/"summary.json").exists()}
    main = summaries["all"]
    validation = read(RESULT / "validation" / "summary.json")
    manifest = read(RESULT / "manifest.json")
    diagnostics = real_diagnostics()
    initial = main["initialization"]
    pipeline = read(RESULT / "validation" / "pipeline_checks.json")
    if not validation["passed"] or not pipeline["passed"]:
        raise ValueError("验证未全部通过，不能生成通过验收的报告")
    tests = validation["tests"]
    means = validation["mean_metrics_across_seeds"]
    au = manifest["data_audit"]
    settings = manifest["settings"]
    test_count, pipeline_count = len(tests), len(pipeline["checks"])
    sim_count = len(validation["simulations"])
    def covariance_text():
        lengths = ", ".join(str(x) for x in au["covariance_block_unique_lengths"])
        return (f"全零 {au['zero_covariance_rows']} 帧，其中连续前缀 {au['leading_zero_covariance_rows']} 帧；"
                f"{au['covariance_blocks']} 块，块长度取值为 {lengths} 帧")
    def metric(value, digits=6):
        return "无同刻样本" if value is None else f"{value:.{digits}f}"
    cn_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
    if cn_path.exists():
        pdfmetrics.registerFont(TTFont("CN", str(cn_path), subfontIndex=0))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    cn = "CN" if cn_path.exists() else "STSong-Light"
    pdfmetrics.registerFont(TTFont("Formula", findfont("DejaVu Sans")))
    pdfmetrics.registerFontFamily(cn, normal=cn, bold=cn, italic=cn, boldItalic=cn)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyCN", fontName=cn, fontSize=9.2, leading=15,
                              spaceAfter=7, wordWrap="CJK", textColor=colors.HexColor("#243447")))
    styles.add(ParagraphStyle("HeadCN", parent=styles["BodyCN"], fontSize=16, leading=23,
                              spaceBefore=8, spaceAfter=14, textColor=colors.HexColor("#0072B2")))
    styles.add(ParagraphStyle("SubCN", parent=styles["BodyCN"], fontSize=11, leading=17,
                              spaceBefore=8, spaceAfter=8, textColor=colors.HexColor("#0072B2")))
    styles.add(ParagraphStyle("SmallCN", parent=styles["BodyCN"], fontSize=8, leading=12))
    styles.add(ParagraphStyle("FormulaText", fontName="Formula", fontSize=9, leading=15,
                              leftIndent=10, spaceBefore=2, spaceAfter=7))
    styles.add(ParagraphStyle("CellCN", parent=styles["BodyCN"], fontSize=8.0, leading=12, spaceAfter=0))
    story = []
    # 中文字体没有覆盖部分上下标/数学符号；逐字符切换到嵌入的数学字体。
    cn_chars = getattr(pdfmetrics.getFont(cn).face, "charWidths", {})
    math_chars = pdfmetrics.getFont("Formula").face.charWidths
    def fallback(text):
        return "".join(f'<font name="Formula">{c}</font>'
                       if ord(c) > 127 and ord(c) not in cn_chars and ord(c) in math_chars else c
                       for c in text)
    def p(text, style="BodyCN"):
        story.append(Paragraph(fallback(text) if style != "FormulaText" else text, styles[style]))
    def h(text):
        p(text, "HeadCN")
    def sub(text):
        p(text, "SubCN")
    def formula(text):
        # 简式公式使用真正下标；运行命令保持逐字可复制。
        if not text.startswith("python "):
            text = re.sub(r"([A-Za-zδωθ]+)_([A-Za-zθ0]+(?:,0)?)",
                          r"\1<sub>\2</sub>", text)
        p(text, "FormulaText")
    def table(rows, widths=None):
        wrapped = [[Paragraph(fallback(escape(str(c))), styles["CellCN"]) for c in row] for row in rows]
        tab = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8F1F7")),
                                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")]),
                                ("VALIGN", (0,0), (-1,-1), "TOP"),
                                ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
                                ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                                ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#AFC6D8"))]))
        story.extend([tab, Spacer(1, 10)])
    def page():
        story.append(PageBreak())
    def figure(name, width=495):
        path = RESULT / "figures" / (name + ".png")
        if not path.is_file():
            raise FileNotFoundError(f"报告需要图表：{path}；请先运行run.py --comparisons")
        image = Image(str(path))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth, image.drawHeight = width, width * ratio
        story.extend([image, Spacer(1, 8)])
    def vec(values, digits=7):
        return "[" + ", ".join(f"{v:.{digits}g}" for v in values) + "]"

    def finish(filename, title):
        def decoration(can, doc):
            can.saveState()
            can.setFont(cn, 8)
            can.setFillColor(colors.HexColor("#667788"))
            can.drawString(45, 814, title)
            can.drawRightString(A4[0] - 45, 26, str(doc.page))
            can.setStrokeColor(colors.HexColor("#CBD6DF"))
            can.line(45, 803, A4[0] - 45, 803)
            can.restoreState()
        doc = SimpleDocTemplate(str(ROOT / filename), pagesize=A4,
                                leftMargin=45, rightMargin=45, topMargin=53, bottomMargin=44,
                                title=title, author="")
        doc.build(list(story), onFirstPage=decoration, onLaterPages=decoration)
        story.clear()
        print(f"Saved {ROOT / filename}")

    h("题目二：IMU 姿态与位置估计")
    p("验收报告 / 15维误差状态扩展卡尔曼滤波", "SubCN")
    p("完成exam.pdf第4–8页全部姿态必做要求及位置选做。结果已生成，阅读本报告不需要安装环境或运行程序。完整推导与辅助实验见TECHNICAL_APPENDIX.pdf；运行说明见README.md。")
    table([["题目层级", "实现与可查看结果"],
           ["姿态必做", "IMU读取、实际时间间隔、陀螺零偏、四元数预测、位姿观测更新；第4页为Roll/Pitch/Yaw曲线"],
           ["位置选做", "位置、速度、加计零偏、重力补偿与世界系积分；第5页为x/y/z曲线"],
           ["结果数据", "results/all/estimates.csv：姿态、位置、速度、偏置与协方差；列说明见COLUMNS.md"],
           ["验证证据", f"{test_count}项数值检查、{pipeline_count}组流程检查、{sim_count}组独立真值仿真；结果见第6页"]], [85, 420])
    sub("核心代码地图")
    table([["功能", "函数"],
           ["数据读取 / 初始零偏", "data_io.py：load_data / initialize"],
           ["预测 / 状态转移与Qd", "eskf.py：ESKF.predict / discretize"],
           ["残差、K、Joseph更新与重置", "eskf.py：ESKF.update_pose"],
           ["观测时间对齐与回放", "run.py：replay"],
           ["六条必需曲线", "plots.py：real_figures"]], [210, 295])
    p(f"主实验逐条融合有效FAST-LIO位姿，R倍率为1；本次保存 {main['output_rows']:,} 帧，接受 {main['counts']['accepted']:,} 次更新。15维实现包含姿态与陀螺零偏所需的6维误差子系统，姿态6维数学形式见技术附录。")
    p("结果边界：FAST-LIO与预测共享IMU，真实记录没有独立真值。对FAST-LIO的贴合程度不等于实际精度；数值验证支持受测条件下的实现，不能替代真实精度验证。", "SmallCN")
    page()

    h("1  状态、预处理与预测")
    formula("x = (p, v, q, b_g, b_a),   δx = [δp, δv, δθ, δb_g, δb_a]ᵀ")
    p("标称参数16个，误差状态15维。q为Hamilton单位四元数，按xyzw存储；R把IMU机体系向量旋转到世界系，右乘误差满足Rtrue=R Exp([δθ]×)。")
    table([["约定", "取值"],
           ["姿态 / 重力", "R=Rz(yaw)Ry(pitch)Rx(roll)；g=[0,0,−9.81] m/s²"],
           ["输入单位", "gyro：rad/s；acc：g，读取时乘一次9.81"],
           ["时间处理", "十进制秒先解析为整数纳秒；按真实相邻时间差积分"],
           ["位姿参考点", "IMU坐标系在世界系中的位姿；输入语义和适用范围见技术原稿"]], [96, 409])
    sub("初始静止估计")
    p(f"收集到 {initial['start_time_s']:.6f} s 后启动，使用 {initial['calibration_begin_s']:.3f} s 起的 {initial['imu_calibration_rows']} 帧IMU；先检查IMU波动和位姿变化。静止区间用于估计bg与条件加计零偏，初始速度为0。")
    formula("b_g,0 = mean(ω_m),    b_a,0 = mean(a_m) + R_0ᵀg")
    p("加计初值依赖初始姿态与重力，未独立分离倾角误差和加计偏置。初始化IMU不足两帧时明确拒绝，避免输出NaN噪声参数。", "SmallCN")
    sub("中点惯性预测")
    formula("ω = ω_m − b_g,  f = a_m − b_a,  R_mid = R Exp([ωΔt/2]×)")
    formula("a_W = R_mid f + g;   p⁺ = p + vΔt + ½a_WΔt²;   v⁺ = v + a_WΔt")
    formula("q⁺ = q ⊗ Exp_q(ωΔt),   b_g⁺ = b_g,   b_a⁺ = b_a")
    formula("P⁻ = ΦPΦᵀ + Q_d,   Φ ≈ I + FΔt + (FΔt)²/2 + (FΔt)³/6")
    p("F由姿态、比力及偏置误差的一阶传播得到；Qd用3点Gauss积分。偏置标称值在预测中不变，其随机游走仍进入协方差。姿态预测和修正后都保持单位四元数。完整F、G和离散化核验见技术附录。")
    page()

    h("2  观测更新与Q、R的作用")
    formula("r = [p_obs − p, Log(RᵀR_obs)]ᵀ,    H = [I 0 0 0 0; 0 0 I 0 0]")
    p("位置用米、姿态残差用旋转向量弧度；Hθ=I为小创新近似。按题目使用CSV协方差的六个对角项。默认把后三项当roll/pitch/yaw方差，以E映射到右局部旋转扰动，再以右雅可比逆映射到Log残差。")
    formula("M_θ = J_r⁻¹(r_θ) E,   M = diag(I, M_θ),   R_obs = M diag(cov) Mᵀ")
    formula("S = HP⁻Hᵀ + R_obs,    K = P⁻HᵀS⁻¹,    δx̂ = Kr")
    formula("P_J = (I−KH)P⁻(I−KH)ᵀ + KR_obsKᵀ")
    formula("q ← q ⊗ Exp_q(δθ̂),   G_reset = diag(I,I,J_r(δθ̂),I,I)")
    formula("P⁺ = G_reset P_J G_resetᵀ")
    p("代码解线性方程求K；位置、速度和偏置作加性注入，姿态作右乘注入，再重置误差协方差。")
    table([["量", "作用"],
           ["Q / Qd", "预测期间增加过程不确定性：陀螺、加计白噪声与两类偏置随机游走"],
           ["R", "描述位姿观测的不确定性；主实验保持题目给定对角方差，经过坐标映射使用"],
           ["K", "由预测P、观测H与R共同决定修正方向和强度"]], [65, 440])
    sub("本次数据与时间对齐")
    p(f"IMU {au['imu']['rows']:,} 帧，平均 {au['imu']['mean_rate_hz']:.3f} Hz；位姿 {au['pose']['rows']:,} 帧，平均 {au['pose']['mean_rate_hz']:.3f} Hz。协方差审计：{covariance_text()}。")
    p("每条位姿只在自身时间戳更新一次；相邻IMU端点插值到积分中点，需最多一个采样区间缓冲。无效协方差不参与更新；重复、冲突、非有限行与缺口均有明确处理及记录。")
    page()

    h("3  必做结果：Roll / Pitch / Yaw")
    figure("attitude")
    p("灰色虚线为FAST-LIO输入，蓝线为ESKF；浅色区间为初始化等待。横轴为相对秒，纵轴为度。Yaw展开仅用于显示连续转动，不参与滤波。")
    p("三条曲线及弧度/角度数据均已保存。两条曲线接近时虚线可能被覆盖；差异另见trajectory_and_input_difference图。贴合程度不构成独立精度证据。图表文件：results/figures/attitude.png和attitude.pdf。", "SmallCN")
    page()

    h("4  选做结果：x / y / z")
    figure("position")
    p("位置由IMU预测与位姿观测共同估计，单位为米，坐标系沿用输入世界系。对应CSV还保存世界系速度、机体系偏置和15个误差方差。")
    p("图表文件：results/figures/position.png和position.pdf。其余轨迹、零偏、NIS、积分基线、中断与敏感性图均作为附加分析保留。", "SmallCN")
    page()

    h("5  实验结果、验证与使用范围")
    sub("主实验：与输入观测的差异")
    table([["统计项", "本次结果"],
           ["确切同刻比较样本", str(main['input_pose_comparison_rows'])],
           ["三维位置差RMSE", metric(main['input_pose_disagreement_position_rmse_m']) + " m"],
           ["SO(3)姿态差RMSE", metric(main['input_pose_disagreement_attitude_rmse_deg']) + "°"],
           ["P最小特征值 / q最大模误差", f"{main['covariance_min_eigenvalue_all_outputs']:.3e} / {main['quaternion_max_norm_error']:.3e}"]], [240, 265])
    sub("独立验证")
    p(f"{test_count}项数值检查与{pipeline_count}组流程检查通过。数学检查核对旋转方向、雅可比和解析导数；数值检查核对离散化与协方差；流程检查核对时间、单位、异常输入和导出。具体测试标签及结果由验证JSON读取。")
    p(f"{sim_count}个固定种子合成实验中，ESKF平均位置真值RMSE为 {means['eskf']['position_rmse_m']:.5f} m，姿态为 {means['eskf']['attitude_geodesic_rmse_deg']:.5f}°。这些数字仅适用于解析轨迹、独立噪声、已知初始姿态及固定偏置的仿真假设。")
    sub("计算能力与限制")
    p(f"在 {escape(manifest['platform'])} / Python {manifest['python']} 上，含积分基线和逐点P检查的平均回放处理率约 {main['processing_rate_hz_with_checks']:.1f} Hz；滤波加基线的每步p99为 {main['filter_plus_baselines_step_p99_ms']:.3f} ms。计时不含文件读写和绘图，不保证现场端到端实时性。")
    p("FAST-LIO与本滤波器共享IMU，相关性可能使模型协方差偏乐观；按协方差变化更新只是一项抽取对照，不保证统计一致。真实绝对精度和优于FAST-LIO的结论需要独立真值。")
    sub("复现与补充材料")
    p("Windows：安装Python 3.12后双击run_windows.cmd；Linux：bash run_linux.sh（尚未在Linux实跑）。Python 3.12为已验证环境；其他≥3.12且<4版本警告后尝试，不承诺兼容。全部数据、对照实验和技术附录随包保留，无需另行索取即可查看结果。")
    p('依据：exam.pdf第4–8页；<link href="https://arxiv.org/abs/1711.02508" color="#0072B2">Solà，Quaternion kinematics for the error-state Kalman filter</link>；<link href="https://github.com/hku-mars/FAST_LIO" color="#0072B2">FAST-LIO官方仓库</link>。更多来源和数据假设见REPORT.md。', "SmallCN")
    finish("REPORT.pdf", "题目二：验收报告")

    h("技术附录：IMU 姿态与位置估计")
    p("15 维误差状态扩展卡尔曼滤波 / 完整程序、实验结果与原理说明", "SubCN")
    p("依据 exam.pdf 第4–8页，完成姿态必做与位置估计选做。输入为给定IMU和FAST-LIO位姿CSV。输入位姿定义为IMU坐标系在世界坐标系中的位置与方向。")
    table([["交付项", "完成内容"],
           ["算法", "位置、速度、四元数、陀螺仪零偏、加计零偏；15维误差状态；完整预测与更新"],
           ["真实实验", "逐帧融合、协方差变化更新、R×10、R×100、2秒观测中断"],
           ["数值验证", f"{test_count}项数值检查、{pipeline_count}组流程检查、{sim_count}组独立真值仿真"],
           ["输出", "Roll/Pitch/Yaw及x/y/z六条时间曲线；CSV、创新、协方差、偏置和实验JSON"],
           ["复现", "Windows一键入口、Linux脚本、依赖版本、原数据副本及SHA-256"]], [80, 425])
    sub("结果与证据边界")
    p(f"主实验生成 {main['output_rows']:,} 帧，使用 {main['counts']['accepted']:,} 条位姿更新。所有输出协方差的最小特征值为 {main['covariance_min_eigenvalue_all_outputs']:.3e}，四元数模长最大误差为 {main['quaternion_max_norm_error']:.3e}。")
    means = validation["mean_metrics_across_seeds"]
    p(f"三组独立仿真的ESKF平均位置RMSE为 {means['eskf']['position_rmse_m']:.5f} m，姿态测地RMSE为 {means['eskf']['attitude_geodesic_rmse_deg']:.5f}°。这些数值只属于所列仿真假设。")
    p("真实数据没有独立真值。FAST-LIO已使用IMU，因此与它的差异只能解释为输入一致性，不能证明实际定位精度或优于FAST-LIO。标准EKF的独立噪声假设在此并不严格成立，报告没有把模型协方差当作已校准的真实置信区间。")
    p("文档导航：本PDF保留完整方法和辅助结果；REPORT.pdf为验收主报告。REPORT.md是绑定题给数据的可编辑技术原稿。README.md给出运行方式、目录和CSV列说明。", "SmallCN")
    page()

    h("1  约定、姿态表示与李群")
    p("世界系记W，IMU机体系记B。R将B系向量转到W系。采用Hamilton四元数，存储顺序xyzw，q和−q代表同一个旋转。重力取g=[0,0,−9.81] m/s²；这是世界系重力竖直的建模约定。")
    table([["表示", "定义与作用", "注意事项"],
           ["欧拉角", "R=Rz(yaw)Ry(pitch)Rx(roll)，便于显示和解释", "pitch=±90°奇异；不能直接对跨±180°的角度做差"],
           ["旋转矩阵", "RᵀR=I，det(R)=1；用于旋转比力", "9个数表示3个自由度，需要保持正交约束"],
           ["单位四元数", "4个参数满足单位模约束；用于标称姿态积分", "只有3个自由度；加性4维更新易破坏约束"]], [68, 218, 219])
    sub("SO(3) 和 so(3)")
    p("SO(3)是旋转矩阵组成的李群：旋转可复合，但普通矩阵相加不会仍是旋转。so(3)是单位元附近的反对称矩阵空间；用三维小旋转向量表示其局部坐标。帽算子满足 [u]×v=u×v，Exp把小旋转送到群上，Log取局部旋转误差。")
    formula("R<sub>true</sub> = R Exp([δθ]×),   q<sub>true</sub> = q ⊗ Exp<sub>q</sub>(δθ)")
    formula("Exp<sub>q</sub>(φ) = [sin(‖φ‖/2) φ/‖φ‖, cos(‖φ‖/2)]  (xyzw)")
    p("实现使用SciPy的Rotation在小角度、近π及四元数等价符号处稳定处理。角度都以弧度参与计算；Yaw展开仅用于图表。")
    sub("状态空间")
    formula("x = (p, v, q, b<sub>g</sub>, b<sub>a</sub>),   3+3+4+3+3 = 16 parameters")
    formula("δx = [δp, δv, δθ, δb<sub>g</sub>, δb<sub>a</sub>]<super>T</super> ∈ ℝ<super>15</super>")
    p("P是15维误差的协方差，不是对16个标称参数直接建立的加性协方差。完整15维实现已包含姿态必做，无须另外复制一个6维滤波器。")
    sub("事实、推断与建模选择")
    p("输入位姿使用IMU参考点约定；四元数方向和轴一致性另受数据支持。ZYX欧拉顺序、竖直重力、初始速度为零及噪声模型属于明确声明的选择。单姿态加计零偏初始化依赖姿态与重力先验，不能同时独立标定所有因素。")
    page()

    h("2  数据检查、初始化与时间对齐")
    au = manifest["data_audit"]
    table([["检查", "IMU / 位姿数据"],
           ["行数", f"{au['imu']['rows']:,} / {au['pose']['rows']:,}"],
           ["跨度", f"{au['imu']['duration_s']:.6f} s / {au['pose']['duration_s']:.6f} s"],
           ["平均频率", f"{au['imu']['mean_rate_hz']:.4f} Hz / {au['pose']['mean_rate_hz']:.4f} Hz"],
           ["同刻时间戳", f"{au['exact_shared_timestamps']:,} 个；仍使用事件调度处理非同刻观测"],
           ["协方差", covariance_text()],
           ["四元数", f"归一化前模误差最大{au['quaternion_norm_max_deviation']:.3e}；等价符号翻转{au['quaternion_sign_flips']}次"],
           ["缺口", f"IMU最大间隔 {1000*au['imu']['dt_max_s']:.4f} ms；按实际Δt传播"]], [93, 412])
    p("读取时将十进制秒先解析为整数纳秒，再减去第一帧时间；保留原始时间基准文本。加速度g乘9.81转换一次，陀螺仪保持rad/s。重复、非有限值、无效四元数和时间冲突按README所列策略处理，并保存审计。")
    sub("初始静止估计")
    p(f"等待到 {initial['start_time_s']:.9f} s，使用{initial['calibration_begin_s']:.3f}秒起已到达的 {initial['imu_calibration_rows']} 帧IMU及有效位姿。位置最大偏离 {1000*initial['max_position_deviation_m']:.4f} mm，姿态最大偏离 {initial['max_attitude_deviation_deg']:.5f}°，支持此窗口近似静止。")
    formula("b<sub>g,0</sub> = mean(ω<sub>m</sub>),   b<sub>a,0</sub> = mean(a<sub>m</sub>) + R<sub>0</sub><super>T</super>g")
    formula("b<sub>g,0</sub> = " + vec(initial["bg0_rad_s"]) + " rad/s")
    formula("b<sub>a,0</sub> = " + vec(initial["ba0_m_s2"]) + " m/s²")
    p("q₀取有效位姿旋转均值，p₀取位置均值，v₀=0。加计式来自静止比力关系aₘ=−Rᵀg+bₐ；姿态倾角误差会混入该估计。初始P保留位置、速度、姿态与偏置的不确定性，具体15个对角值保存在summary.json。")
    sub("因果时序")
    p("每条位姿在自己的时间戳只更新一次。区间中点IMU由相邻端点插值，允许最多一个IMU周期缓冲；没有用未来位姿插值。初始化完成前不输出在线结果。全零协方差帧不参与更新；全量审计与主循环计数分别记录。")
    page()

    h("3  预测模型与过程协方差")
    p("IMU加速度是含偏置和噪声的比力。补偿偏置后旋转到世界系，再加重力。陀螺仪用于在SO(3)上积分。区间采用中点姿态，最大子步长0.01秒。")
    formula("ω = ω<sub>m</sub> − b<sub>g</sub>,   f = a<sub>m</sub> − b<sub>a</sub>")
    formula("R<sub>mid</sub> = R Exp([ω Δt/2]×),   a<sub>W</sub> = R<sub>mid</sub>f + g")
    formula("p⁺ = p + v Δt + ½a<sub>W</sub>Δt²,   v⁺ = v + a<sub>W</sub>Δt")
    formula("q⁺ = q ⊗ Exp<sub>q</sub>(ω Δt),   b<sub>g</sub>⁺ = b<sub>g</sub>,   b<sub>a</sub>⁺ = b<sub>a</sub>")
    p("标称偏置保持不变，并不意味着偏置不确定性不增长；偏置随机游走由过程噪声进入P。四元数通过单位旋转运算保持归一化。")
    sub("线性误差模型的非零块")
    formula("δẋ = F δx + G<sub>n</sub> n")
    table([["F的行、列块", "数值"], ["p, v", "I"], ["v, θ", "−Rmid [f]×"],
           ["v, ba", "−Rmid"], ["θ, θ", "−[ω]×"], ["θ, bg", "−I"]], [220, 285])
    p("Gₙ把陀螺白噪声、加计白噪声和两类偏置随机游走映射到误差状态；其非零块分别为θ行−I、v行−R、bg行I、ba行I。假设各噪声组及各轴互不相关。")
    formula("P⁻ = Φ P Φ<super>T</super> + Q<sub>d</sub>")
    formula("Φ ≈ I + FΔt + (FΔt)²/2 + (FΔt)³/6")
    formula("Q<sub>d</sub> = ∫₀<super>Δt</super> exp(Fs) G<sub>n</sub> Q<sub>c</sub> G<sub>n</sub><super>T</super> exp(Fs)<super>T</super> ds")
    p("实现用3点Gauss积分近似Qd，并按B(s)B(s)ᵀ累加以保持半正定；这是冻结F的小步长近似。用独立30×30 Van Loan矩阵指数核查：具体测试步长与最大绝对差逐例保存在validation/summary.json。")
    p(f"初始样本标准差乘√Δt得到白噪声幅度的标量均方近似：gyro={initial['gyro_noise_density']:.7g}，acc={initial['acc_noise_density']:.7g}。偏置随机游走幅度为{settings['gyro_bias_rw']:.4g} (rad/s)/√s、{settings['acc_bias_rw']:.4g} (m/s²)/√s。这些是模型参数，未做独立Allan方差标定；加计各轴噪声明显不同，因此各向同性是简化。", "SmallCN")
    page()

    h("4  观测更新、增益与误差重置")
    formula("r = [p<sub>obs</sub> − p,  Log(R<super>T</super>R<sub>obs</sub>)]<super>T</super>")
    p("H的位置块选择δp，姿态块选择δθ，其余为零。姿态块Hθ=I依赖小创新的一阶近似，不声称任意大旋转残差下精确。")
    sub("从题目协方差到残差协方差")
    p("按题面读取cov_00、cov_11、cov_22、cov_33、cov_44、cov_55，单位分别为m²与rad²。题面把姿态项称为欧拉角方差，因此先使用ZYX欧拉微扰到右局部旋转微扰的映射E：")
    table([["E的第1行", "1", "0", "−sin(pitch)"],
           ["第2行", "0", "cos(roll)", "sin(roll) cos(pitch)"],
           ["第3行", "0", "−sin(roll)", "cos(roll) cos(pitch)"]], [86, 60, 150, 209])
    formula("M<sub>θ</sub> = J<sub>r</sub>⁻¹(r<sub>θ</sub>) E,   M = blockdiag(I, M<sub>θ</sub>)")
    formula("R<sub>obs</sub> = M diag(σ²<sub>x</sub>,σ²<sub>y</sub>,σ²<sub>z</sub>,σ²<sub>roll</sub>,σ²<sub>pitch</sub>,σ²<sub>yaw</sub>) M<super>T</super>")
    p("Jᵣ是SO(3)右雅可比。若输入元数据另行规定后三项为局部旋转向量方差，可使用rotvec分支去掉E。默认遵照题面，不擅自更改协方差语义；交叉项按题目要求舍去。")
    sub("卡尔曼更新")
    formula("S = H P⁻ H<super>T</super> + R<sub>obs</sub>,   K = P⁻ H<super>T</super> S⁻¹,   δx̂ = K r")
    formula("P<sub>J</sub> = (I−KH)P⁻(I−KH)<super>T</super> + K R<sub>obs</sub> K<super>T</super>")
    p("代码通过解线性方程求K，不显式求逆；Joseph形式改善数值稳定性。加性注入p、v和两类偏置，姿态用右乘注入q←q⊗Expq(δθ̂)。")
    formula("G<sub>reset</sub> = blockdiag(I,I,J<sub>r</sub>(δθ̂),I,I)")
    formula("P⁺ = G<sub>reset</sub> P<sub>J</sub> G<sub>reset</sub><super>T</super>,   δx ← 0")
    p("Q决定预测期间不确定性如何增长，R决定观测的相对可信度，K据两者决定修正程度。不能为追求曲线贴合而把R置零。主实验R倍数为1；R×10、R×100仅是显式敏感性对照。")
    p("NIS=rᵀS⁻¹r作为诊断输出。默认不据NIS剔除真实LIO修正；同源相关、噪声失配使其不具备严格卡方置信度解释。", "SmallCN")
    page()

    h("5  独立验证与仿真实验")
    tests = validation["tests"]
    p(f"{test_count}项数值检验全部通过，另有{pipeline_count}组端到端流程检查通过。检查实际模型行为，不以重复实现公式代替独立参照。")
    table([["检验", "结果"]] + [[test["label"], "通过" if test["passed"] else "失败"]
          for test in tests], [415, 90])
    p("流程检查涵盖纳秒时间戳、列别名、单位换算、重复冲突、四元数符号、错开2ms的异步观测、每条观测只消费一次、完全停更和CSV回读。")
    sub("可核查的独立真值实验")
    p("使用解析平滑位置和欧拉函数构造真值及其导数，不调用被测滤波器生成真值。IMU为200Hz，位姿观测10Hz且噪声独立；20秒轨迹，前2秒静止校准，12–13.5秒停止位姿更新。初始静止位姿已知，偏置为固定已知值，随机种子为7、29、113。")
    table([["方法", "平均位置RMSE (m)", "平均姿态测地RMSE (°)"]] +
          [[label, f"{means[key]['position_rmse_m']:.5f}", f"{means[key]['attitude_geodesic_rmse_deg']:.5f}"] for key, label in
           [("raw_imu", "原始IMU积分"), ("bias_corrected_imu", "初始零偏校正积分"), ("eskf", "ESKF")]], [175, 165, 165])
    p("以上是三种子的各自RMSE再取算术均值，不是大量Monte Carlo实验的置信区间。位置误差按三维欧氏距离计算，姿态误差为Log(RtrueᵀRest)的模；不能把仿真数字外推为真实FAST-LIO数据精度。")
    page()

    h("6  真实数据结果与局限")
    p(f"输出从 {initial['start_time_s']:.9f} s 开始，共 {main['output_rows']:,} 帧；与输入位姿比较只使用 {main['input_pose_comparison_rows']:,} 个确切同刻样本，不用未来观测补齐。")
    rows = [["实验", "更新数", "位置差RMSE (m)", "姿态差RMSE (°)", "平均NIS"]]
    for key, name in [("all", "按题逐帧"), ("cov_change", "协方差变化"), ("R_x10", "R×10"), ("R_x100", "R×100"), ("dropout", "2秒中断")]:
        if key in summaries:
            s = summaries[key]
            rows.append([name, s["counts"]["accepted"], metric(s["input_pose_disagreement_position_rmse_m"]),
                         metric(s["input_pose_disagreement_attitude_rmse_deg"]), metric(s["nis_mean"], 3)])
    table(rows, [94, 58, 125, 135, 93])
    p("表中全是与输入观测的差异。R越小通常越贴近输入，但贴合不能证明更准确。较大的NIS提示模型与观测统计不完全匹配；不能靠调R使曲线好看就宣称模型已经校准。")
    sub("同源与时间相关性")
    interval = au["covariance_change_interval_s"]["mean"]
    p(f"协方差块切换间隔均值为 {metric(interval, 9)} s。变化帧更新减少使用观测的次数，但仍未消除同一IMU导致的相关性；仅凭块结构不能证明观测独立或统计更稳健。原记录更详细的审计及适用范围见REPORT.md。")
    sub("数值与性能")
    p(f"主实验所有输出P的最小特征值 {main['covariance_min_eigenvalue_all_outputs']:.6g}，最大不对称量 {main['covariance_max_asymmetry']:.3g}；q模最大偏差 {main['quaternion_max_norm_error']:.3g}。真实数据未出现数值发散。")
    p(f"记录环境为Python {manifest['python']}，{escape(manifest['platform'])}。本次含基线及逐点P检查的回放耗时 {main['wall_time_s_with_checks']:.3f} s，吞吐约 {main['processing_rate_hz_with_checks']:.1f} Hz。统计不含文件导出和绘图；它支持此机上的计算能力判断，不构成硬实时调度保证。")
    sub("未被数据消除的限制")
    p("没有独立位置/姿态真值；运动主要是平面和Yaw，不能据此验证强三维激励下的全部偏置可观性；重力与加计初值耦合；Q采用标量白噪声近似；题面欧拉协方差与具体FAST-LIO导出语义可能有差别。任何实际精度或置信区间声明都需要进一步标定和独立参考。")
    page()

    h("7  必需输出：Roll / Pitch / Yaw")
    figure("attitude")
    p("灰色虚线为FAST-LIO输入观测，蓝线为15维ESKF。浅色区域为初始化等待。所有滤波运算使用四元数；显示时Yaw展开以保留完整转圈，避免±180°换支导致假跳变。")
    p("三条时间曲线及相应弧度/角度数据已导出。姿态测地差异与模型不确定性另见 results/figures/trajectory_and_input_difference、innovation_and_model_uncertainty。", "SmallCN")
    page()
    h("8  选做输出：x / y / z")
    figure("position")
    p("位置单位为米，世界系沿用输入位姿定义；曲线重合处虚线可能被覆盖，差异另见trajectory_and_input_difference图。速度、加计偏置和位置协方差一并保存在结果CSV中。")
    p("三条曲线完成题目的位置输出要求；二维轨迹、纯惯导对照、偏置和2秒观测中断图均作为辅助结果单独提供。", "SmallCN")
    page()
    h("9  独立合成真值：代表性轨迹")
    figure("synthetic_validation")
    p("图示种子7。浅橙区间表示位姿观测中断，ESKF继续依靠IMU传播，恢复后重新融合。下左图单独展示ESKF误差，避免被纯积分的大漂移掩盖。全部种子逐帧真值及估计均保存在results/validation，可重算RMSE。")
    p("该实验验证已知模型、独立观测和特定噪声下的实现行为；与真实数据的FAST-LIO同源观测有本质统计差别。", "SmallCN")
    page()
    h("10  验收对应、运行与来源")
    table([["题目要求", "实现与交付"],
           ["读取、时间间隔、零偏、异常处理", "data_io.py；run.py事件时序；manifest与初始参数JSON"],
           ["姿态与位置预测、偏置随机游走", "eskf.py / predict；15维误差顺序符合题面"],
           ["P预测、Q设计、四元数归一化", "三阶Phi、Gauss噪声积分、单位Rotation运算"],
           ["位姿观测、CSV协方差、时间对齐", "update_pose；rpy映射；主实验R倍数1；逐条事件更新"],
           ["残差、K、状态及P更新", "李代数残差、线性求解、Joseph形式、右乘注入与重置"],
           ["Roll/Pitch/Yaw与x/y/z", "results/all/estimates.csv；attitude/position PNG及PDF"],
           ["一组实验及简单原理说明", "真实对照和独立仿真；REPORT.pdf、技术附录与REPORT.md"],
           ["可复现完整程序", "README、依赖版本、Windows/Linux入口、数据副本、数值与流程验证"]], [210, 295])
    sub("重新运行")
    p("Windows双击run_windows.cmd。脚本使用项目内隔离环境，缺失时通过Python 3.12创建并安装锁定依赖。Linux可在Python 3.12环境运行bash run_linux.sh；Linux脚本已提供，但未在Linux系统实跑。")
    formula("python validate.py --output results/validation")
    formula("python validate_pipeline.py --output results/validation/pipeline_checks.json")
    formula("python run.py --comparisons")
    formula("python build_report.py")
    sub("来源与材料身份")
    p("题目来源：exam.pdf，第4–8页；实验输入为题给imu.csv及pose_cov.csv。原始数据副本的SHA-256保存在manifest.json。输入位姿参考点采用IMU原点约定。所有图表由程序从实际结果生成，仿真文件明确标记为synthetic。")
    p('理论参考：Joan Solà, Quaternion kinematics for the error-state Kalman filter (2017). <link href="https://arxiv.org/abs/1711.02508" color="#0072B2">arxiv.org/abs/1711.02508</link>。用于核对旋转约定与误差状态方法，本文公式按本项目状态顺序展开。', "SmallCN")
    p('FAST-LIO实现及观测来源背景：<link href="https://github.com/hku-mars/FAST_LIO" color="#0072B2">github.com/hku-mars/FAST_LIO</link>。官方当前源码不能证明这份录制使用的版本，故未据此擅自修改数据约定。', "SmallCN")
    page()
    h("11  姿态六维形式与统计诊断")
    sub("姿态必做的数学基线")
    p("仅保留姿态与陀螺零偏时，标称状态为(q,bg)，误差δz=[δθ,δbg]ᵀ。右局部误差的连续模型如下，I和0均为3×3块：")
    formula("F₆ = [−[ω]×  −I;  0  0],   G₆ = [−I  0;  0  I]")
    formula("Q_c,6 = diag(σ_g²I, σ_bg²I),   H₆ = [I  0]")
    formula("P₆⁻ = Φ₆P₆Φ₆ᵀ + Q_d,6,   G_reset,6 = diag(J_r(δθ̂), I)")
    p("Qd,6由六维连续模型积分得到；姿态残差与协方差映射沿用前文。这里给出独立姿态模型的数学形式。完整程序采用15维联合估计，位置观测通过交叉协方差影响姿态，不能仅把p、v、ba置零就等价为六维滤波器。")
    sub("NIS量级：失配存在，原因尚未辨识")
    nall, nwindow = diagnostics["nis_all"], diagnostics["nis_window_65_105_s"]
    ratio = nall["mean"] / 6 if nall["mean"] is not None else None
    p(f"主实验平均NIS为 {metric(nall['mean'], 4)}。零均值且创新协方差正确时，六维模型的名义期望为6；本次比值为 {metric(ratio, 4)}。65–105秒窗口均值为 {metric(nwindow['mean'], 4)}，99%分位数为 {metric(nwindow['p99'], 4)}，最大值为 {metric(nwindow['max'], 4)}。统计从保存的创新CSV重新计算。")
    p("候选原因包括Q/R失配、未建模同源交叉相关与时间相关、潜在时间偏差或其他模型误差。没有独立同步标定，不能指定时间偏差为主要原因；在IMU原点位姿约定下，也不能直接认定缺少杆臂补偿。")
    p("不确定性图绘制一维右局部旋转误差的标准差，并非严格的Euler yaw标准差。三维更新后姿态输入差与六维更新前NIS也不是同一种统计量，不能直接作比值衡量“过度自信几倍”。")
    sub("偏置与曲线的读法")
    p("偏置状态可吸收弱激励下的未建模效应，不能直接当作独立测得的物理零偏变化。无器件规格或温度记录时，不能断言短时变化不可能来自传感器。姿态和位置曲线贴合处可能遮住输入虚线；差异图单独显示这些变化。")
    p("完整诊断数值：results/all/diagnostics.json。该分析不改写题面R、不调整时间戳、不引入未确认外参，且不构成真实精度证明。", "SmallCN")
    finish("TECHNICAL_APPENDIX.pdf", "题目二：技术附录")


if __name__ == "__main__":
    make_report()
