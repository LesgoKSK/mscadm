from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import build_current_progress_plain_docx as base


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TEMPLATE = REPORTS / "MS-CADM_论文方法与复现专题报告.docx"
OUTPUT = REPORTS / "CURRENT_RESEARCH_PROGRESS_ZERO_BACKGROUND.docx"
ASSET_DIR = REPORTS / "assets" / "progress_zero"

IMAGES = [
    (ASSET_DIR / "01_task.png", "研究任务：从天气预报生成 100 个可能的明日风电剧本。"),
    (ASSET_DIR / "02_problem.png", "旧模型的问题：平均位置尚可，但范围过窄、小时变化不自然。"),
    (ASSET_DIR / "03_journey.png", "研究路线：每一步都在回答一个问题，并排除一个不可靠答案。"),
    (ASSET_DIR / "04_family.png", "Flow 与 Diffusion 的公平比较：各有长处，没有总冠军。"),
    (ASSET_DIR / "05_idea.png", "当前创新思路：天气和去噪阶段都合适时，才开启小幅时间修正。"),
    (ASSET_DIR / "06_probe.png", "下一步小体检：48 个轻量实验决定保留双开关、单开关还是停止。"),
]


def add_figure(
    out: list[str],
    image_rels: list[tuple[str, Path]],
    index: int,
    *,
    width_inches: float = 6.15,
) -> None:
    path, caption = IMAGES[index - 1]
    rid = f"rId{19 + index}"
    image_rels.append((rid, path))
    out.append(base.picture(rid, path, caption, index, width_inches=width_inches))


def simple_card(title: str, text: str, *, fill: str, color: str) -> str:
    parts = [
        base.run(title, bold=True, color=color, size=22),
        '<w:r><w:br/></w:r>',
        base.run(text, color=color, size=20),
    ]
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="9026" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
        '</w:tblPr><w:tblGrid><w:gridCol w:w="9026"/></w:tblGrid><w:tr><w:trPr><w:cantSplit/></w:trPr>'
        + base.cell(parts, 9026, fill=fill, align="left")
        + '</w:tr></w:tbl>' + base.p("", after=80)
    )


def build_body() -> tuple[str, list[tuple[str, Path]]]:
    out: list[str] = []
    image_rels: list[tuple[str, Path]] = []

    # Cover
    out.append(base.p("", before=1500, after=0))
    out.append(base.p(
        [base.run("ZERO-BACKGROUND · GROUP MEETING", bold=True, color=base.GOLD, size=19)],
        style="ReportKicker", align="center", after=320,
    ))
    out.append(base.p([
        base.run("我们在教模型写", bold=True, color=base.NAVY, size=52),
        '<w:r><w:br/></w:r>',
        base.run("100 个靠谱的明日风电剧本", bold=True, color=base.NAVY, size=52),
    ], style="ReportTitle", align="center", after=260, line=680, keep_lines=True))
    out.append(base.p(
        [base.run("风电联合场景生成阶段进展 · 零基础大白话版", color=base.BLUE, size=29)],
        style="ReportSubtitle", align="center", after=300,
    ))
    out.append(base.p(
        [base.run("不要求懂扩散模型，不要求懂概率预测，不要求先读论文", color=base.MUTED, size=20)],
        align="center", after=100,
    ))
    out.append(base.p(
        [base.run("整理日期：2026-08-28　｜　建议汇报 12–15 分钟", color=base.MUTED, size=18)],
        align="center", after=400,
    ))
    out.append(base.callout(
        "先把任务讲明白，再讲哪里坏了、我们排除了什么、下一步怎样用最小实验决定创新点。",
        label="这份文档只做一件事", fill=base.LIGHT_BLUE, color=base.DARK_BLUE,
    ))
    out.append(base.p(
        [base.run("说明：这是 validation 阶段的研究进展，不是最终测试集结论。", italic=True, color=base.MUTED, size=18)],
        align="center", before=180, after=0,
    ))
    out.append(base.page_break())

    # One-minute orientation
    out.append(base.heading("如果只听一分钟，请记住这三句话", 1))
    out.append(simple_card(
        "第一句：我们不是只预测一个数。",
        "我们要生成 100 条可能的 24 小时风电轨迹，让电网提前看到不同风险。",
        fill=base.LIGHT_BLUE, color=base.DARK_BLUE,
    ))
    out.append(simple_card(
        "第二句：我们已经找到真正的痛点。",
        "旧模型的平均位置不算离谱，但 100 条轨迹太挤、太抖，前后小时关系不像真实风电。",
        fill=base.AMBER_LIGHT, color=base.AMBER,
    ))
    out.append(simple_card(
        "第三句：下一步不是直接堆一个大模型。",
        "先用 48 个小实验找出“什么天气、生成到哪一步”最需要时间信息，再决定最终结构。",
        fill=base.GREEN_LIGHT, color=base.GREEN,
    ))
    out.append(base.callout(
        "现在最有价值的成果不是“某个模型赢了”，而是我们已经知道下一次实验必须回答什么。",
        label="阶段判断", fill=base.LIGHT, color=base.NAVY,
    ))
    out.append(base.page_break())

    # Task
    out.append(base.heading("一、我们到底在做什么", 1))
    out.append(base.plain(
        "假设明天要安排电网。天气预报告诉我们风可能怎么吹，但风电不会只按一条固定曲线走。电网真正需要的是：明天有哪些合理的可能走法？最坏会怎样？要留多少备用？"
    ))
    add_figure(out, image_rels, 1)
    out.append(base.heading("用一个生活比喻", 2))
    out.append(base.plain(
        "普通预测像导航只给你一条“最可能路线”。场景生成像同时给出 100 条可能路线，并告诉你哪些路可能堵、哪些路风险大。我们生成的不是地图上的路线，而是 10 个风电区域未来 24 小时的功率轨迹。"
    ))
    out.append(base.table(
        ["模型看到什么", "模型交出什么", "为什么有用"],
        [["明天的风速、风向和区域信息", "100 个可能的 24 小时风电剧本", "做备用、调度和风险准备"]],
        [3000, 3000, 3026],
    ))
    out.append(base.callout(
        "一条剧本必须把 10 个区域和 24 个小时放在同一个世界里，不能把互相矛盾的片段随便拼起来。",
        label="最关键的要求",
    ))
    out.append(base.page_break())

    # Problem
    out.append(base.heading("二、旧模型到底坏在哪里", 1))
    out.append(base.plain(
        "最开始我们也容易被一个“平均分”迷惑：平均功率差不多，就以为模型不错。后来把 100 条轨迹画出来、把相邻小时拆开看，才发现它只是中心位置凑合，作为一组未来剧本并不可靠。"
    ))
    add_figure(out, image_rels, 2)
    out.append(base.heading("三个问题，翻译成人话", 2))
    out.append(base.numbered(1, "平均位置还行：大方向可能没偏太远。"))
    out.append(base.numbered(2, "范围太窄：模型说自己很确定，真实结果却经常跑到 100 条剧本之外。"))
    out.append(base.numbered(3, "轨迹太抖：这一小时升、下一小时猛降，像把电影帧顺序弄乱。"))
    out.append(base.callout(
        "论文写的是 90% 预测区间，本地复现实际只覆盖约 36%。也就是说，模型严重低估了“明天可能有多不确定”。",
        label="最直观的报警", fill=base.RED_LIGHT, color=base.RED,
    ))
    out.append(base.page_break())

    # Journey
    out.append(base.heading("三、我们为什么做了这么多实验", 1))
    out.append(base.plain(
        "因为不能一看到问题就随便换网络。每次实验都要回答一个具体问题：是论文没复现？是数据串了？是随机种子碰巧？是时间模块真的有用？还是只是把曲线抹平了？"
    ))
    add_figure(out, image_rels, 3)
    out.append(base.table(
        ["阶段", "它回答的问题", "答案"],
        [
            ["复现论文", "原论文方法在本地能否得到同样结果？", "程序跑通，但核心数值没对上"],
            ["拆开诊断", "问题主要在平均值，还是轨迹怎么走？", "主要短板在轨迹动态"],
            ["重建考场", "比较是否被数据和随机性搅乱？", "无泄漏、三种子框架已稳定"],
            ["测试时间模块", "时间顺序真的有用吗？", "有用，但现有做法不稳定"],
            ["比较两条路线", "Flow 或 Diffusion 是否全面更好？", "各有长处，没有总冠军"],
        ],
        [1900, 4200, 2926],
    ))
    out.append(base.page_break())

    # Results in plain words
    out.append(base.heading("四、目前最重要的实验结果", 1))
    out.append(base.heading("结果 1：论文方法重做出来了，但论文成绩没有重现", 2))
    out.append(base.plain(
        "我们把 MS-CADM 的数据、模型、训练、采样和评价链都搭起来了。程序能正常训练，也能生成场景；但本地核心 CRPS 是 0.1074，论文报告 0.0873，本地高约 23%。这里 CRPS 越低越好。"
    ))
    out.append(base.callout(
        "准确说法是“忠实重实现没有复现论文 headline”，不是“已经证明论文错误”。因为论文没有公开足够细的代码和参数。",
        label="不要说过头",
    ))
    out.append(base.heading("结果 2：真正需要修的是“怎么走”", 2))
    out.append(base.plain(
        "旧联合模型和逐区 DDPM 的平均水平几乎打平；但旧联合模型更容易出现过度抖动和不真实的大跳变。天气越是变速、转向或区域差异大，这个问题越明显。"
    ))
    out.append(base.heading("结果 3：公平实验的地基已经搭好", 2))
    out.append(base.plain(
        "我们把已经看过的日期、数据填充边界、天气编码器、随机噪声和验证库都锁住。换三个随机种子后，R0 的分数波动明显缩小，正式通过稳定性门。"
    ))
    out.append(base.callout(
        "像三位同学参加同一场考试：试卷、考场、时间都一样，只允许“初始手气”不同。这样才能知道模型是否稳定。",
        label="为什么要三种子", fill=base.GREEN_LIGHT, color=base.GREEN,
    ))
    out.append(base.page_break())

    # Temporal mechanism
    out.append(base.heading("五、时间模块：明明有改善，为什么仍然不采用", 1))
    out.append(base.heading("先说好消息：模型确实读懂了一部分时间顺序", 2))
    out.append(base.plain(
        "我们做了一个很关键的反证：把 1、2、3、4 小时的顺序故意打乱。如果所谓“时间模块”仍然同样好，它可能只是把曲线变平，并没有理解前后关系。结果是正常顺序明显好于打乱顺序，所以真实时间信号确实存在。"
    ))
    out.append(base.heading("再说坏消息：它的副作用还太大", 2))
    out.append(base.plain(
        "T1-feature 和 T1-source 都改善了相邻小时变化和前后关系，但换随机种子后，平均分布或整体联合质量会波动；其中随机源版本还把内部相关系数几乎顶到上限。"
    ))
    out.append(base.table(
        ["候选", "它做对了什么", "为什么停下"],
        [
            ["T1-feature", "正常顺序比 Shuffle 好；时间关系改善", "平均/联合安全门和种子稳定性未过"],
            ["T1-source", "时间关系改善更大", "跨种子更飘，相关系数接近饱和"],
        ],
        [2200, 3500, 3326],
    ))
    out.append(base.callout(
        "“时间信息有用”不等于“这个时间模块可以上线”。就像药有效，但副作用和个体差异太大，仍然不能批准。",
        label="这一步最容易误解", fill=base.AMBER_LIGHT, color=base.AMBER,
    ))
    out.append(base.page_break())

    # Family comparison
    out.append(base.heading("六、Flow 和 Diffusion：到底选谁", 1))
    out.append(base.plain(
        "这两种方法都从随机噪声出发，最后生成未来轨迹。Flow 像连续运输带；Diffusion 像把模糊照片一步步擦清。我们让它们使用同一数据、同一计算预算和同一验证随机库。"
    ))
    add_figure(out, image_rels, 4)
    out.append(base.heading("正式结果", 2))
    out.append(base.bullet("Flow：平均分布、整体联合质量和覆盖率更好。", bold_prefix="Flow："))
    out.append(base.bullet("Diffusion：前后时间关系更好。", bold_prefix="Diffusion："))
    out.append(base.bullet("相邻小时升降：两者几乎一样。", bold_prefix="相邻小时升降："))
    out.append(base.callout(
        "正式 winner 为空。后续拿 Diffusion 做研究载体，只因为它有清楚的“模糊—清晰”阶段，方便研究时间信息何时介入；不是因为它已经战胜 Flow。",
        label="结论", fill=base.AMBER_LIGHT, color=base.AMBER,
    ))
    out.append(base.page_break())

    # Idea
    out.append(base.heading("七、现在准备研究的创新点", 1))
    out.append(base.plain(
        "过去的时间模块像一个一直抢方向盘的辅助驾驶：平静天气也开，画面还很模糊时也开，结果有时帮忙、有时添乱。现在的想法是给它两个开关，而且修正幅度必须有上限。"
    ))
    add_figure(out, image_rels, 5)
    out.append(base.heading("两个开关分别问什么", 2))
    out.append(base.numbered(1, "天气开关：今天的风是否正在明显变速、转向，或不同区域变化很不一样？"))
    out.append(base.numbered(2, "阶段开关：当前这一步去噪，是否已经能看见时间结构、又还来得及修？"))
    out.append(base.plain(
        "只有两个答案都偏向“是”时，时间模块才做一个小修正；其余时候让稳定基础模型自己工作。"
    ))
    out.append(base.callout(
        "创新不在于“首次把 SSM/GRU 塞进 Diffusion”，而在于测清并利用“天气状态 × 去噪阶段”的时间价值差异。",
        label="论文主张应该聚焦这里", fill=base.GREEN_LIGHT, color=base.GREEN,
    ))
    out.append(base.page_break())

    # Next probe
    out.append(base.heading("八、下一步具体做什么", 1))
    out.append(base.plain(
        "我们不会马上写完整双开关大模型。先把稳定的 Diffusion 基座冻结，只训练很小、从零开始、修正幅度受限的时间插件。这样如果分数变好，原因更容易说清。"
    ))
    add_figure(out, image_rels, 6)
    out.append(base.heading("48 个小实验是怎么来的", 2))
    out.append(base.table(
        ["拆分", "数量", "为什么"],
        [
            ["去噪阶段", "8 段", "看时间信息在哪一段最有价值"],
            ["时间顺序", "正常 / 打乱", "排除“只是平滑”这种假改善"],
            ["随机种子", "3 个", "排除一次走运"],
            ["总数", "8 × 2 × 3 = 48", "都是轻量插件，不是 48 个大模型"],
        ],
        [2600, 1800, 4626],
    ))
    out.append(base.heading("实验结束后只允许三种决定", 2))
    out.append(base.numbered(1, "天气差异和阶段差异都稳定存在：做双开关。"))
    out.append(base.numbered(2, "只存在一种差异：只留一个开关，不强行复杂化。"))
    out.append(base.numbered(3, "没有稳定规律，或打乱顺序同样好：停止这条路线。"))
    out.append(base.page_break())

    # Status and discussion
    out.append(base.heading("九、现在能说什么，不能说什么", 1))
    out.append(base.heading("现在可以肯定地说", 2))
    for text in [
        "MS-CADM 的完整工程链已经重做，但核心数值和不确定性覆盖没有复现。",
        "旧模型的主要短板是轨迹动态，不只是平均功率。",
        "正常时间顺序比打乱时间顺序更有效，说明时间信息是真信号。",
        "Flow 与 Diffusion 各有长处，目前没有全面胜者。",
        "下一步的 48 个轻量实验能够直接决定创新结构是否值得做。",
    ]:
        out.append(base.bullet(text))
    out.append(base.heading("现在绝对不能说", 2))
    for text in [
        "不能说 Diffusion 已经赢了 Flow。",
        "不能说已经找到最终时间模块。",
        "不能把 validation 结果说成最终测试或外部泛化。",
        "不能为了论文故事，在探针结果不支持时仍强行做双开关。",
    ]:
        out.append(base.bullet(text))
    out.append(base.callout(
        "我们已经从“哪个模型分数高一点”推进到“知道故障在哪、知道哪些办法不可靠、知道下一次实验必须回答什么”。",
        label="当前阶段性成果",
    ))

    out.append(base.heading("十、组会现场可以直接照着说", 1))
    out.append(base.callout(
        "我们做的不是预测一条风电曲线，而是给明天准备 100 个可能剧本。复现论文后发现，模型虽然平均位置还行，但剧本太窄、小时变化不真实。我们用无泄漏、三种子和打乱时间顺序的对照证明：时间信息确实有用，但现有时间模块副作用太大。Flow 和 Diffusion 的公平比较也没有总冠军。因此下一步不急着堆模型，而是用 48 个小实验找出时间信息在什么天气、去噪哪一步最有用；只有结果支持，才做双开关模型。",
        label="60 秒总结", fill=base.LIGHT_BLUE, color=base.DARK_BLUE,
    ))
    out.append(base.heading("希望大家讨论三个问题", 2))
    out.append(base.numbered(1, "8 段 × 正常/打乱 × 3 种子的“小体检”，是否足以支撑后续模型设计？"))
    out.append(base.numbered(2, "主创新是否应明确聚焦“天气状态 × 去噪阶段的选择性时间干预”？"))
    out.append(base.numbered(3, "边界 0/1 状态的 Atom-v2，应该放在本工作后半部分，还是另开一条研究线？"))
    out.append(base.page_break())

    # Appendix
    out.append(base.heading("附录 A：术语翻译表（汇报时不必全讲）", 1))
    out.append(base.table(
        ["术语", "大白话"],
        [
            ["Scenario / 场景", "一种可能发生的完整未来剧本"],
            ["Ramp", "下一小时比这一小时升多少或降多少"],
            ["Lagged relation", "这一小时的变化与下一小时是否衔接得像真的"],
            ["Coverage90", "标为 90% 的预测范围，实际圈住真实结果的比例"],
            ["Flow", "把随机噪声沿一条连续路线变成未来轨迹"],
            ["Diffusion", "把随机噪声像模糊照片一样一步步擦清"],
            ["SNR / 去噪阶段", "当前画面有多模糊或多清楚"],
            ["Shuffle", "故意打乱小时顺序，检查模型是否真的理解时间"],
            ["Seed / 随机种子", "换一次初始手气，检查结果是不是碰巧"],
            ["Atom", "必须精确等于 0 或 1 的特殊状态"],
        ],
        [2600, 6426],
    ))
    out.append(base.heading("附录 B：关键数字（被追问时再展示）", 1))
    out.append(base.table(
        ["问题", "关键数字", "解释"],
        [
            ["论文复现", "CRPS 0.1074 vs 0.0873", "本地高约 23%，未复现 headline"],
            ["复现覆盖", "coverage90 = 0.3595", "名义 90%，实际约 36%，严重过窄"],
            ["旧模型轨迹", "总变差 +80.33%；最大跳变 +57.13%", "平均接近不代表路径合理"],
            ["时间 feature", "ramp 改善 0.002162；lagged VS 改善 19.9%", "有信号，但正式安全门未过"],
            ["时间 source", "ramp 改善 0.002697；lagged VS 改善 62.2%", "收益更大，但跨种子更不稳"],
            ["Flow vs DDPM", "level 0.079661 vs 0.080484", "Flow 较好"],
            ["Flow vs DDPM", "lagged VS 0.017720 vs 0.017189", "DDPM 较好"],
            ["下一步规模", "8 × 2 × 3 = 48", "轻量 adapter 体检"],
        ],
        [2300, 3000, 3726],
    ))
    out.append(base.heading("附录 C：仓库中的原始证据", 1))
    for item in [
        "reports/MSCADM_PAPER_REPRODUCTION_REPORT.md —— 论文复现和数值审计",
        "reports/CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md —— 轨迹动态与天气条件诊断",
        "reports/ARCHITECTURE_V1_R0_STABILITY_V2_3_FORMAL_RESULT.md —— 三种子稳定性",
        "reports/ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md —— 时间模块与 Shuffle 裁决",
        "reports/ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md —— Flow vs DDPM 正式比较",
        "reports/CURRENT_RESEARCH_PROGRESS_GROUP_MEETING_STANDALONE.html —— 详细浏览器版汇报",
    ]:
        out.append(base.bullet(item))
    out.append(base.p([base.run("文档结束", bold=True, color=base.GOLD, size=18)], align="center", before=300, after=0))

    return "".join(out), image_rels


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>风电联合场景生成阶段进展：零基础大白话版</dc:title>
 <dc:subject>风电场景生成组会汇报</dc:subject><dc:creator>multi_scaleCADM</dc:creator>
 <cp:keywords>wind power; scenario generation; plain language; diffusion; flow</cp:keywords>
 <dc:description>面向零基础听众的 WPS 兼容图文版阶段汇报，无 Word 公式对象。</dc:description>
 <cp:lastModifiedBy>multi_scaleCADM</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''


def footer_xml(blank: bool = False) -> str:
    if blank:
        body = base.p("", after=0)
    else:
        field = (
            base.run("风电场景生成阶段进展 · 零基础版　｜　第 ", color=base.MUTED, size=16)
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
        zout.writestr("word/header2.xml", base.header_xml("风电联合场景生成阶段进展 · 零基础版").encode("utf-8"))
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
