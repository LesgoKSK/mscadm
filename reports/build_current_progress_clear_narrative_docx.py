from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import build_current_progress_plain_docx as base


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TEMPLATE = REPORTS / "MS-CADM_论文方法与复现专题报告.docx"
OUTPUT = REPORTS / "CURRENT_RESEARCH_PROGRESS_CLEAR_NARRATIVE.docx"
CLEAR_ASSETS = REPORTS / "assets" / "progress_clear"

IMAGES = [
    (
        CLEAR_ASSETS / "01_task_formal.png",
        "研究任务：在 NWP 条件下生成多区域、多时段联合风电场景，并同时评价边际校准、时间变化、联合依赖和边界状态。",
    ),
    (
        REPORTS / "assets" / "mscadm_paper" / "reproduction_scorecard.png",
        "MS-CADM 复现结果总览：本地重实现能够训练和采样，但核心概率指标与论文报告值存在明显差距。",
    ),
    (
        ROOT / "outputs" / "cross_model_diagnostics" / "primary_paired_differences.png",
        "跨模型配对诊断：边际 level CRPS 接近，但 ramp、总变差与最大跳变暴露出明显的轨迹动态差距。",
    ),
    (
        ROOT / "outputs" / "cross_model_diagnostics" / "primary_nwp_regime_ramp.png",
        "NWP 条件分组诊断：变速、转向和空间异质天气下的 ramp 差距明显大于平静天气。",
    ),
    (
        CLEAR_ASSETS / "02_evidence_chain.png",
        "研究推进逻辑：复现、诊断、公平协议、时间机制、模型家族比较与效用探针形成连续证据链。",
    ),
    (
        CLEAR_ASSETS / "03_family_formal.png",
        "family-v1.1 正式比较：Flow 在边际/联合质量与覆盖率上占优，DDPM 在滞后依赖上占优，未产生正式胜者。",
    ),
    (
        CLEAR_ASSETS / "04_hypothesis_formal.png",
        "候选创新假设：在稳定 v-DDPM 基座上，由天气动态性与去噪阶段共同控制有界时间残差。",
    ),
    (
        CLEAR_ASSETS / "05_probe_formal.png",
        "Temporal Utility Probe：8 个 log-SNR 区间、正常/打乱时间顺序和 3 个训练种子构成 48 个轻量 retained run。",
    ),
]


def add_figure(
    out: list[str],
    image_rels: list[tuple[str, Path]],
    index: int,
    *,
    width_inches: float = 6.10,
) -> None:
    path, caption = IMAGES[index - 1]
    rid = f"rId{19 + index}"
    image_rels.append((rid, path))
    out.append(base.picture(rid, path, caption, index, width_inches=width_inches))


def section_answer(text: str) -> str:
    return base.callout(text, label="本节结论", fill=base.LIGHT_BLUE, color=base.DARK_BLUE)


def build_body() -> tuple[str, list[tuple[str, Path]]]:
    out: list[str] = []
    image_rels: list[tuple[str, Path]] = []

    # Cover
    out.append(base.p("", before=1420, after=0))
    out.append(base.p(
        [base.run("RESEARCH PROGRESS · CLEAR NARRATIVE", bold=True, color=base.GOLD, size=19)],
        style="ReportKicker", align="center", after=320,
    ))
    out.append(base.p([
        base.run("风电联合场景生成", bold=True, color=base.NAVY, size=52),
        '<w:r><w:br/></w:r>',
        base.run("阶段研究进展", bold=True, color=base.NAVY, size=52),
    ], style="ReportTitle", align="center", after=260, line=680, keep_lines=True))
    out.append(base.p(
        [base.run("从 MS-CADM 复现、跨模型诊断到选择性时间干预假设", color=base.BLUE, size=28)],
        style="ReportSubtitle", align="center", after=310,
    ))
    out.append(base.p(
        [base.run("GEFCom2014 Wind · 10 个区域 · 24 小时 · 每个预测日 100 个联合场景", color=base.MUTED, size=19)],
        align="center", after=90,
    ))
    out.append(base.p(
        [base.run("整理日期：2026-08-31　｜　组会汇报建议 18–22 分钟", color=base.MUTED, size=18)],
        align="center", after=370,
    ))
    out.append(base.callout(
        "本文档按照“研究问题—已有证据—阶段决策—下一步验证”的顺序组织，正文保留必要术语并在首次出现时解释，详细数值集中在表格与附录。",
        label="文档定位", fill=base.LIGHT_BLUE, color=base.DARK_BLUE,
    ))
    out.append(base.p(
        [base.run("说明：当前结论来自 train/validation 阶段；selection、calibration 与最终外部测试仍保持封存。", italic=True, color=base.MUTED, size=18)],
        align="center", before=160, after=0,
    ))
    out.append(base.page_break())

    # Executive summary
    out.append(base.heading("摘要：当前研究处于什么阶段", 1))
    out.append(base.plain(
        "本项目关注多区域风电联合场景生成：模型不仅要预测各区域未来功率的边际分布，还要生成具有合理时间演化、跨区域依赖和精确边界状态的完整联合样本。研究最初以 MS-CADM 复现为起点，随后因核心概率指标未复现而转向结构化诊断。"
    ))
    out.append(base.plain(
        "跨模型诊断表明，现有联合模型的主要不足不是平均功率水平，而是相邻小时变化、滞后依赖以及动态天气条件下的轨迹质量。为避免数据泄漏和随机性混杂，项目随后建立 Architecture-v1 公平实验协议，并完成三种子稳定性验证。"
    ))
    out.append(base.plain(
        "时间机制实验进一步证明：真实时间顺序相对于 Shuffle 对照确实包含可利用信号；但 T1-feature 与 T1-source 均未同时满足非劣性和跨种子稳定性要求，因此正式判为 No-Go。Flow 与 v-prediction Joint DDPM 的公平比较同样没有产生全面胜者。"
    ))
    out.append(base.plain(
        "因此，当前研究不再预设“给 Diffusion 增加某个 SSM/GRU 模块”就是答案，而是提出可证伪的选择性时间干预假设：时间信息的价值可能同时依赖 NWP 动态状态和 Diffusion 去噪阶段。下一步将通过 Temporal Utility Probe 先测量这种效用分布，再决定是否实现双门控、单门控或终止该路线。"
    ))
    out.append(base.table(
        ["已经完成", "当前结论", "尚未完成"],
        [[
            "复现、跨模型诊断、公平协议、时间机制对照、Flow–DDPM 正式比较",
            "时间信号成立，但现有候选不稳定；模型家族无正式胜者",
            "Temporal Utility Probe、门控结构裁决、封存数据上的最终验证",
        ]],
        [3000, 3100, 2926],
    ))
    out.append(base.page_break())

    # Background and task
    out.append(base.heading("一、研究背景与任务定义", 1))
    out.append(base.heading("1. 为什么需要场景生成", 2))
    out.append(base.plain(
        "风电具有明显的不确定性。点预测只给出一条期望轨迹，无法充分支持备用容量、机组启停和风险评估。场景生成则从条件分布中抽取多条可能轨迹，用样本集合近似未来结果的范围、形状和联合依赖。"
    ))
    out.append(base.heading("2. 本研究中的“联合”含义", 2))
    out.append(base.plain(
        "每个场景成员同时包含 10 个区域未来 24 小时的功率。所谓联合场景，不是先独立生成 10 条区域轨迹再任意拼接，而是要求同一成员中的所有区域和时刻共同对应一种一致的未来演化。"
    ))
    add_figure(out, image_rels, 1)
    out.append(base.heading("3. 评价为什么不能只看一个总分", 2))
    out.append(base.table(
        ["评价维度", "所回答的问题", "代表指标"],
        [
            ["边际准确性与校准", "每个区域、每个小时的概率分布是否准确且不过窄", "level CRPS、coverage、width"],
            ["相邻小时变化", "升降方向和变化幅度是否合理", "ramp CRPS、total variation"],
            ["时空联合依赖", "跨区域、跨时刻的相关结构是否接近真实数据", "joint ES、lagged VS"],
            ["精确边界状态", "精确 0/1 状态的概率、次数与持续时间是否合理", "atom Brier、duration"],
        ],
        [2200, 4300, 2526],
    ))
    out.append(section_answer(
        "研究目标不是单纯降低边际误差，而是在统一联合样本中同时保持校准、动态结构、时空依赖和边界语义。"
    ))
    out.append(base.page_break())

    # Starting paper
    out.append(base.heading("二、为什么从 MS-CADM 复现开始", 1))
    out.append(base.plain(
        "MS-CADM 是直接面向风电条件场景生成的扩散模型工作，研究任务、数据形式和多尺度 NWP 条件编码均与本项目高度相关。因此，它适合作为方法理解、工程基线和问题发现的起点。复现的目的不是将该结构预设为最终架构，而是建立一个可运行、可诊断的参考系统。"
    ))
    out.append(base.heading("1. 已重建的工程链", 2))
    for text in [
        "数据链：GEFCom2014 Wind 的 10 个区域、731 个完整日历日和逐小时 NWP 条件。",
        "模型链：多尺度条件编码、AdaLN Transformer、噪声与方差预测。",
        "训练链：26,000 次更新、条件遮蔽、扩散损失与 checkpoint 管理。",
        "生成链：250 步 ancestral 采样，并对照不同 DDIM 步数。",
        "评价链：MAE、RMSE、CRPS、QS、ES、VS、coverage 与区间宽度。",
    ]:
        out.append(base.bullet(text))
    out.append(base.heading("2. 复现结果与主要差距", 2))
    out.append(base.plain(
        "本地重实现能够完成训练和采样，但论文核心概率指标未重现。论文报告 CRPS 为 0.0873，本地结果为 0.1074，后者高约 23.02%；名义 90% 区间的实际 coverage 仅为 0.3595，说明生成场景严重欠离散，即真实结果经常落在样本范围之外。"
    ))
    add_figure(out, image_rels, 2, width_inches=6.05)
    out.append(base.callout(
        "由于原文未公开完整代码，且模型宽度、噪声日程、损失权重、mask 粒度与 checkpoint 选择等实现细节并不完整，严谨结论是“忠实重实现未复现 headline”，而不是据此断言论文结果错误。",
        label="解释边界", fill=base.AMBER_LIGHT, color=base.AMBER,
    ))
    out.append(section_answer(
        "复现完成了方法链重建，但数值差距和严重欠覆盖说明：仅继续微调复现模型不足以回答后续研究问题，需要进一步定位误差结构。"
    ))
    out.append(base.page_break())

    # Diagnosis
    out.append(base.heading("三、从总分比较转向跨模型结构诊断", 1))
    out.append(base.heading("1. 为什么拆分 ramp、cross-lag、atom 和 NWP regime", 2))
    out.append(base.plain(
        "level CRPS 主要反映单个位置的概率分布；即使它表现接近，两套模型生成的完整轨迹仍可能具有完全不同的平滑性、持续性和跨区域协调方式。因此，诊断进一步检查相邻小时变化（ramp）、滞后依赖（cross-lag）、精确边界状态持续时间（atom duration），并按 NWP 动态条件分组。"
    ))
    out.append(base.heading("2. 边际水平接近，但轨迹动态差距明显", 2))
    out.append(base.plain(
        "旧 mixed-measure 联合模型与逐区 product DDPM 的配对结果显示，level CRPS 基本持平；但旧联合模型的 ramp CRPS 高约 5.19%，total variation CRPS 高约 80.33%，最大绝对 ramp CRPS 高约 57.13%。这表明主要差距集中在轨迹如何随时间演化，而不是整体功率水平。"
    ))
    add_figure(out, image_rels, 3, width_inches=6.10)
    out.append(base.heading("3. 空间同时关系较好，时间滞后关系较弱", 2))
    out.append(base.plain(
        "旧联合模型能够较好保持同一时刻的跨区域耦合，但 lag-1 附近的时间依赖误差明显较大。例如，同一区域 lag-1 相关误差在旧联合模型中为 0.3002，而逐区 DDPM 为 0.0149。由此可见，空间联合建模本身并未自动解决时间前后关系。"
    ))
    out.append(base.heading("4. 差距在动态天气条件下扩大", 2))
    out.append(base.plain(
        "平静天气下，两类模型的 ramp 差距约为 0.00186；在风速快速变化、风向明显转动和空间异质性较强时，差距扩大到约 0.00372、0.00360 和 0.00355。该现象提示：时间机制的价值可能依赖天气动态状态，而不是在所有日期上均匀存在。"
    ))
    add_figure(out, image_rels, 4, width_inches=6.05)
    out.append(base.callout(
        "这些旧档案已被永久标记为 R-SEEN，只用于形成研究假设，不用于最终模型选择或泛化声明。",
        label="证据等级", fill=base.AMBER_LIGHT, color=base.AMBER,
    ))
    out.append(section_answer(
        "后续创新应直接针对时间动态，并重点检验动态天气条件；仅优化边际损失或继续增加空间模块，无法从诊断结果中得到充分支持。"
    ))
    out.append(base.page_break())

    # Protocol
    out.append(base.heading("四、Architecture-v1：先建立可比较的实验基础", 1))
    out.append(base.plain(
        "早期实验中，训练种子变化会同时影响初始化、条件编码器、日期顺序、噪声和边界分配，难以判断性能波动来自模型结构还是实验随机性。Architecture-v1 的目标是锁定无关变量，使候选模型之间尽量只保留一个受检因素。"
    ))
    out.append(base.heading("1. 数据角色与防泄漏规则", 2))
    for text in [
        "已经参与旧诊断的日期永久标记为 R-SEEN，不再作为未见数据。",
        "缺失值填充不得跨越 train、validation、calibration 和 selection 边界。",
        "validation 用于架构开发；selection、calibration 与最终测试保持封存。",
    ]:
        out.append(base.bullet(text))
    out.append(base.heading("2. 共同控制条件", 2))
    for text in [
        "共享 NWP encoder、Atom 模块和精确边界语义。",
        "共享 10 区域 × 24 小时联合输出、场景成员数和评价脚本。",
        "共享训练日期顺序、验证随机库、Atom allocation 和初始噪声。",
        "尽量匹配参数量、训练预算、早停规则和采样 NFE。",
    ]:
        out.append(base.bullet(text))
    out.append(base.heading("3. R0 三种子稳定性", 2))
    out.append(base.table(
        ["指标", "旧 seed spread", "修复后 spread", "变化"],
        [
            ["level CRPS", "0.003645", "0.001331", "下降 63.5%"],
            ["ramp CRPS", "0.001464", "0.000422", "下降 71.1%"],
            ["joint ES 相对 spread", "3.737%", "1.510%", "下降 59.6%"],
        ],
        [2500, 2100, 2100, 2326],
    ))
    out.append(section_answer(
        "R0 已通过正式稳定性门，说明后续时间机制与模型家族比较建立在可复核、低混杂的共同协议上。"
    ))

    # Evidence chain figure at transition
    out.append(base.heading("五、从复现到当前假设的完整逻辑", 1))
    out.append(base.plain(
        "前述工作并非若干互相独立的模型试验，而是一条连续证据链：复现暴露校准问题，跨模型诊断定位时间动态短板，公平协议排除实验混杂，时间机制对照验证时间信号，模型家族比较确定不存在简单的全面胜者，最终将问题收敛为“时间信息在何种条件下有价值”。"
    ))
    add_figure(out, image_rels, 5)
    out.append(base.page_break())

    # Temporal mechanism
    out.append(base.heading("六、时间机制实验：信号成立，但候选正式 No-Go", 1))
    out.append(base.heading("1. v3.0：自由时间随机源导致场景塌缩", 2))
    out.append(base.plain(
        "第一版时间随机源能够自由改变均值、相关性和尺度。虽然训练损失下降，但 coverage90 降至 0.507，区间宽度缩至 0.171，梯度裁剪比例达到 47.2%。该候选通过压缩场景离散度降低训练难度，破坏了概率输出的校准性，因此立即判为 No-Go。"
    ))
    out.append(base.heading("2. v3.1：限制为方差保持的相关结构", 2))
    out.append(base.plain(
        "修订版本仅允许学习前后小时相关系数，不额外学习均值和尺度，使随机源方差保持在 1 附近。单种子试验表明，feature recurrence 与 source recurrence 均可能改善 ramp 和滞后依赖，但单种子结果不足以支持正式结论。"
    ))
    out.append(base.heading("3. v3.2：三种子与 Shuffle 正式对照", 2))
    out.append(base.table(
        ["候选", "ramp 改善", "lagged VS 改善", "Chronological vs Shuffle", "裁决"],
        [
            ["T1-feature", "0.002162", "19.9%", "正常顺序明显更好", "No-Go"],
            ["T1-source", "0.002697", "62.2%", "正常顺序明显更好", "No-Go"],
        ],
        [1850, 1500, 1700, 2300, 1676],
    ))
    out.append(base.plain(
        "Chronological 均优于对应 Shuffle，说明收益确实来自时间顺序，而非单纯增加参数或统一平滑。然而，T1-feature 未通过 level/joint 非劣门与 joint seed spread；T1-source 的跨种子波动更明显，内部相关系数还在约 99.8% 的位置趋于饱和。"
    ))
    out.append(section_answer(
        "可以确认“时间顺序包含有效信息”，但不能确认“现有 recurrence 机制适合作为最终模块”。收益与副作用仍然耦合，因此两个候选均正式停止。"
    ))
    out.append(base.page_break())

    # Family comparison
    out.append(base.heading("七、Flow 与 Joint DDPM：为什么没有直接选出赢家", 1))
    out.append(base.heading("1. 先解决 DDPM 的 sampler contract 故障", 2))
    out.append(base.plain(
        "family-v1 中的 epsilon-prediction DDPM 在训练阶段数值正常，但采样时有 89.28% 的内部坐标被推到精确 0/1，连续潜变量幅度达到约正负 12,000。该问题属于生成语义失效，而不是一般的指标劣化，因此 family-v1 被冻结为 D0 sampler contract No-Go。"
    ))
    out.append(base.plain(
        "family-v1.1 将 DDPM 参数化改为 v-prediction，并重新通过 P0、三种子训练、checkpoint resume 和采样契约检查。正式比较复用 3 个 F0 best EMA 与 3 个 D0-v best EMA，每个 checkpoint 使用 3 个采样 seed，共形成 18 个共同 validation 场景档案。"
    ))
    add_figure(out, image_rels, 6)
    out.append(base.heading("2. 指标权衡", 2))
    out.append(base.plain(
        "Flow 的 level CRPS、joint ES 和 coverage90 更好；DDPM 的 lagged VS 更好；ramp CRPS 几乎相同。DDPM 的 width90 更窄，但 coverage 同时下降，因此不能将更窄区间单独解释为优势。双方均未满足预注册的全面支配条件。"
    ))
    out.append(base.heading("3. 为什么后续使用 Diffusion 作为研究载体", 2))
    out.append(base.plain(
        "该选择不是模型家族胜负裁决。新的研究问题涉及去噪阶段：Diffusion 的每个训练时刻都对应明确噪声强度和 log-SNR，便于测量时间信息在不同生成阶段的边际效用。Flow 仍保留为外部参考基线，并在后续正式比较中继续出现。"
    ))
    out.append(section_answer(
        "模型家族比较的正式结果是“无胜者”。Diffusion 仅因实验可识别性更适合承载下一阶段假设，而不是因为其总体性能已经优于 Flow。"
    ))
    out.append(base.page_break())

    # Hypothesis
    out.append(base.heading("八、当前候选创新：选择性时间干预", 1))
    out.append(base.heading("1. 假设来源", 2))
    out.append(base.plain(
        "跨 NWP regime 诊断表明，时间动态缺口在变速、转向和空间异质天气下更明显；v3.2 又表明正常时间顺序具有真实效用，但全天气、全阶段持续启用的时间机制会损害稳定性。两组证据共同支持一个更具体的假设：时间干预的收益可能同时依赖天气状态和去噪阶段。"
    ))
    add_figure(out, image_rels, 7)
    out.append(base.heading("2. 模型组成与各自作用", 2))
    out.append(base.table(
        ["组成", "作用", "研究定位"],
        [
            ["稳定 v-DDPM 基座", "建模主体条件分布", "工程基础，不作为主要创新"],
            ["天气门 g_met", "根据 train-only NWP dynamicity 判断时间干预需求", "检验天气交互"],
            ["阶段门 g_snr", "根据 log-SNR 判断当前去噪阶段的时间效用", "检验阶段结构"],
            ["有界时间残差", "零初始化、小参数量、限制修正幅度", "降低破坏基础分布的风险"],
            ["Atom 模块", "处理精确 0/1/interior 状态", "探针阶段冻结，另行研究 Atom-v2"],
        ],
        [2100, 4100, 2826],
    ))
    out.append(base.heading("3. 当前 Atom 建模边界", 2))
    out.append(base.plain(
        "现有 Atom head 对每个区域、每个时刻输出三分类概率，再通过 shared-priority allocation 提供成员间共同性。它并不是完整学习出的 10 区域 × 24 小时联合离散过程；训练使用真实状态，推理使用预测和分配状态，也存在配置分布差异。"
    ))
    out.append(base.plain(
        "但 Temporal Utility Probe 中不应同时修改 Atom 和时间机制，否则即使性能改善也难以识别来源。因此探针阶段冻结当前 Atom，Atom-v2 作为独立诊断方向保留。"
    ))
    out.append(section_answer(
        "主创新应聚焦“天气状态 × 去噪阶段”的时间效用差异；v-diffusion、normalization、Min-SNR 和 mixed-measure 仅作为稳定基础或系统组成，不能单独声称创新。"
    ))
    out.append(base.page_break())

    # Next probe
    out.append(base.heading("九、下一阶段：Temporal Utility Probe", 1))
    out.append(base.heading("1. 为什么先做探针，而不是直接训练完整 M3", 2))
    out.append(base.plain(
        "完整双门控模型会同时引入阶段门、天气门、时间 residual 和额外优化自由度。若直接训练，即使最终分数提高，也难以判断收益来自哪一部分。探针采用冻结基座和小型 adapter，将问题转化为可识别的局部效用测量。"
    ))
    add_figure(out, image_rels, 8)
    out.append(base.heading("2. retained 实验矩阵", 2))
    out.append(base.plain(
        "将训练 timestep 映射为 8 个 log-SNR 区间；每个区间分别训练 Chronological 与固定 Shuffle adapter；每项使用 3 个训练种子。总规模为 8 × 2 × 3 = 48 个 retained 轻量 run。所有 run 共享验证日期、噪声库、Atom allocation、基础 checkpoint 和评价脚本。"
    ))
    out.append(base.heading("3. 预注册性能门", 2))
    out.append(base.table(
        ["检查项", "预期门槛", "目的"],
        [
            ["ramp CRPS", "改善至少 0.001，且配对区间支持", "确认相邻小时动态收益"],
            ["lagged VS", "相对改善至少 5%", "确认滞后依赖收益"],
            ["level / joint", "恶化不超过 0.0015 / 2%", "防止以边际或联合质量为代价"],
            ["coverage / width", "差不超过 0.02 / 增幅不超过 10%", "保护概率校准"],
            ["SNR 结构", "至少两个相邻区间形成稳定峰", "决定是否保留阶段门"],
            ["天气交互", "dynamic 收益稳定高于 stable", "决定是否保留天气门"],
            ["Shuffle 对照", "Chronological 必须更好", "确认收益来自真实时间顺序"],
        ],
        [2100, 3700, 3226],
    ))
    out.append(base.heading("4. 结果对应的决策", 2))
    out.append(base.numbered(1, "SNR 结构与天气交互均成立：进入完整双门控模型。"))
    out.append(base.numbered(2, "仅一种效应稳定成立：只实现对应单门控。"))
    out.append(base.numbered(3, "效应平坦、跨种子不稳或 Shuffle 等效：终止门控路线。"))
    out.append(section_answer(
        "下一阶段的目标不是获得一个更高的 validation 分数，而是识别时间信息效用是否具有稳定的阶段结构和天气交互，从而决定后续结构是否有证据基础。"
    ))
    out.append(base.page_break())

    # Status and meeting close
    out.append(base.heading("十、阶段性结论与组会讨论重点", 1))
    out.append(base.heading("1. 当前可以成立的结论", 2))
    for text in [
        "MS-CADM 方法链已完成工程重构，但核心概率指标与覆盖率未复现。",
        "旧联合模型的主要缺口集中在 ramp、lag-1 与动态天气条件，而非单纯边际水平。",
        "Architecture-v1 已显著降低三种子波动，为正式比较提供了共同基础。",
        "Chronological 优于 Shuffle，说明时间顺序包含真实可利用信息。",
        "现有 T1-feature 与 T1-source 未满足稳定性和非劣要求，正式 No-Go。",
        "Flow 与 D0-v DDPM 各有相对优势，但不存在正式全面胜者。",
        "Temporal Utility Probe 是当前信息增益最高、且能够否证候选假设的下一实验。",
    ]:
        out.append(base.bullet(text))
    out.append(base.heading("2. 当前不能作出的结论", 2))
    for text in [
        "不能宣称 Diffusion 已经优于 Flow。",
        "不能宣称现有时间 recurrence 就是最终模型。",
        "不能将 validation 改善解释为最终测试或外部泛化结果。",
        "不能在探针证据不支持时仍预设双门控结构成立。",
        "不能声称当前 Atom 模块已经学习完整联合离散过程。",
    ]:
        out.append(base.bullet(text))
    out.append(base.heading("3. 建议组会重点讨论", 2))
    out.append(base.numbered(1, "8-bin × Chronological/Shuffle × 3-seed 的探针规模是否足以支持模型结构决策？"))
    out.append(base.numbered(2, "论文主贡献是否应明确限定为选择性时间干预，而将 v-diffusion 与 mixed-measure 降为基础？"))
    out.append(base.numbered(3, "Atom-v2 应作为同一工作中的支撑模块，还是拆分为独立研究方向？"))
    out.append(base.callout(
        "本项目已从“复现一个已有模型”推进到“基于诊断和负结果提出可检验的新假设”。下一步的关键不是立即增加模型复杂度，而是先确认时间效用是否具有稳定的天气依赖和去噪阶段结构。",
        label="汇报收束",
    ))
    out.append(base.page_break())

    # Appendices
    out.append(base.heading("附录 A：指标解释", 1))
    out.append(base.table(
        ["指标", "含义", "期望方向"],
        [
            ["level CRPS", "单个区域、单个小时的概率分布与真实值之间的距离", "越低越好"],
            ["ramp CRPS", "相邻小时功率差分分布与真实变化之间的距离", "越低越好"],
            ["joint ES", "将 10 区域 × 24 小时视为整体后的联合样本质量", "越低越好"],
            ["lagged VS", "跨时刻、跨区域差异结构与真实依赖的偏差", "越低越好"],
            ["coverage90", "真实值落入名义 90% 场景区间的比例", "应接近 0.90"],
            ["width90", "名义 90% 场景区间宽度", "须与 coverage 联合解释"],
            ["atom Brier", "0 / 1 / interior 三种状态概率的误差", "越低越好"],
            ["seed spread", "改变网络初始化后指标的离散程度", "越小越稳定"],
        ],
        [1900, 5200, 1926],
    ))
    out.append(base.heading("附录 B：证据文件索引", 1))
    for item in [
        "reports/MSCADM_PAPER_REPRODUCTION_REPORT.md —— MS-CADM 方法链、复现结果与解释边界",
        "reports/CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md —— ramp、cross-lag、atom 与 NWP regime 诊断",
        "reports/ARCHITECTURE_V1_R0_STABILITY_V2_3_FORMAL_RESULT.md —— R0 三种子稳定性正式结果",
        "reports/ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md —— 时间机制与 Shuffle 正式裁决",
        "reports/ARCHITECTURE_V1_FAMILY_V1_1_TRAINING_RESULT.md —— v-pred 修订与 retained 训练",
        "reports/ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md —— Flow vs DDPM 正式比较",
        "reports/CURRENT_RESEARCH_PROGRESS_GROUP_MEETING_STANDALONE.html —— 浏览器详细图文版",
    ]:
        out.append(base.bullet(item))
    out.append(base.p([base.run("文档结束", bold=True, color=base.GOLD, size=18)], align="center", before=320, after=0))

    return "".join(out), image_rels


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>风电联合场景生成阶段研究进展：清晰叙事版</dc:title>
 <dc:subject>从 MS-CADM 复现到选择性时间干预假设</dc:subject><dc:creator>multi_scaleCADM</dc:creator>
 <cp:keywords>wind power; scenario generation; diffusion; rectified flow; temporal utility</cp:keywords>
 <dc:description>以研究问题、证据、决策和下一步验证为主线的 WPS 兼容图文汇报。</dc:description>
 <cp:lastModifiedBy>multi_scaleCADM</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''


def footer_xml(blank: bool = False) -> str:
    if blank:
        body = base.p("", after=0)
    else:
        field = (
            base.run("风电联合场景生成阶段研究进展　｜　第 ", color=base.MUTED, size=16)
            + '<w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:color w:val="66737F"/><w:sz w:val="16"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'
            + base.run(" 页", color=base.MUTED, size=16)
        )
        body = base.p([field], style="Footer", align="center", after=0)
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="{base.NS["w"]}" xmlns:r="{base.NS["r"]}">{body}</w:ftr>'


def build() -> tuple[Path, str]:
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    for path, _caption in IMAGES:
        if not path.exists():
            raise FileNotFoundError(path)

    body, image_rels = build_body()
    document = base.document_xml(body)
    rels = base.rels_xml(image_rels)
    excluded = {
        "word/document.xml",
        "word/_rels/document.xml.rels",
        "docProps/core.xml",
        "docProps/app.xml",
        "word/header1.xml",
        "word/footer1.xml",
        "word/header2.xml",
        "word/footer2.xml",
    }
    with zipfile.ZipFile(TEMPLATE, "r") as zin, zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename in excluded or info.filename.startswith("word/media/"):
                continue
            zout.writestr(info, zin.read(info.filename))
        zout.writestr("word/document.xml", document.encode("utf-8"))
        zout.writestr("word/_rels/document.xml.rels", rels.encode("utf-8"))
        zout.writestr("docProps/core.xml", core_xml().encode("utf-8"))
        zout.writestr("docProps/app.xml", base.app_xml().encode("utf-8"))
        zout.writestr("word/header1.xml", base.header_xml().encode("utf-8"))
        zout.writestr("word/footer1.xml", footer_xml(blank=True).encode("utf-8"))
        zout.writestr("word/header2.xml", base.header_xml("风电联合场景生成阶段研究进展 · 清晰叙事版").encode("utf-8"))
        zout.writestr("word/footer2.xml", footer_xml().encode("utf-8"))
        for index, (_rid, path) in enumerate(image_rels, 1):
            zout.writestr(f"word/media/plain_progress_{index}.png", path.read_bytes())

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    return OUTPUT, digest


if __name__ == "__main__":
    output, digest = build()
    print(f"output={output}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={digest}")
