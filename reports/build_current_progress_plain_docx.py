from __future__ import annotations

import hashlib
import struct
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TEMPLATE = REPORTS / "MS-CADM_论文方法与复现专题报告.docx"
OUTPUT = REPORTS / "CURRENT_RESEARCH_PROGRESS_PLAIN_LANGUAGE.docx"

IMAGES = [
    (
        ROOT / "reports/assets/mscadm_paper/reproduction_scorecard.png",
        "复现总览：本地结果与论文结果的差距。概率指标的偏差明显大于点预测指标。",
    ),
    (
        ROOT / "outputs/cross_model_diagnostics/primary_paired_differences.png",
        "跨模型诊断：平均功率接近，但 ramp、最大跳变和 lag-1 结构差距明显。",
    ),
    (
        ROOT / "outputs/cross_model_diagnostics/pooled_cross_lag_error.png",
        "不同时间间隔下的相关结构误差。旧联合模型在 lag=1 附近暴露出明显短板。",
    ),
    (
        ROOT / "outputs/cross_model_diagnostics/primary_nwp_regime_ramp.png",
        "天气越是快速变速、明显转向或空间差异大，旧联合模型的 ramp 缺口越明显。",
    ),
]

NAVY = "17365D"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
GOLD = "C27C0E"
TEXT = "263238"
MUTED = "66737F"
LIGHT = "F6F8FA"
LIGHT_BLUE = "EAF2F8"
GREEN = "2E7D32"
GREEN_LIGHT = "EAF6ED"
AMBER = "B26B00"
AMBER_LIGHT = "FFF4DC"
RED = "B23A48"
RED_LIGHT = "FDECEF"
WHITE = "FFFFFF"
LINE = "CBD4DD"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def x(value: object) -> str:
    return escape(str(value), quote=True)


def run(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    color: str = TEXT,
    size: int | None = None,
    font: str = "Microsoft YaHei",
) -> str:
    props = [
        f'<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="{font}"/>',
        f'<w:color w:val="{color}"/>',
    ]
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if size is not None:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t{preserve}>{x(text)}</w:t></w:r>'


def p(
    parts: str | list[str] = "",
    *,
    style: str | None = None,
    align: str | None = None,
    before: int | None = None,
    after: int = 120,
    line: int = 360,
    left: int | None = None,
    first_line: int | None = None,
    hanging: int | None = None,
    keep_next: bool = False,
    keep_lines: bool = False,
    page_break_before: bool = False,
    shade: str | None = None,
) -> str:
    if isinstance(parts, str):
        parts = [run(parts)] if parts else []
    ppr: list[str] = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    spacing = [f'w:after="{after}"', f'w:line="{line}"', 'w:lineRule="auto"']
    if before is not None:
        spacing.append(f'w:before="{before}"')
    ppr.append(f'<w:spacing {" ".join(spacing)}/>')
    indent: list[str] = []
    if left is not None:
        indent.append(f'w:left="{left}"')
    if first_line is not None:
        indent.append(f'w:firstLine="{first_line}"')
    if hanging is not None:
        indent.append(f'w:hanging="{hanging}"')
    if indent:
        ppr.append(f'<w:ind {" ".join(indent)}/>')
    if keep_next:
        ppr.append("<w:keepNext/>")
    if keep_lines:
        ppr.append("<w:keepLines/>")
    if page_break_before:
        ppr.append("<w:pageBreakBefore/>")
    if shade:
        ppr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>')
    return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{"".join(parts)}</w:p>'


def plain(text: str, *, bold: bool = False, color: str = TEXT, size: int | None = None) -> str:
    return p([run(text, bold=bold, color=color, size=size)], keep_lines=True)


def heading(text: str, level: int = 1) -> str:
    return p([run(text)], style=f"Heading{level}", keep_next=True)


def bullet(text: str, *, level: int = 0, bold_prefix: str | None = None) -> str:
    left = 420 + level * 360
    parts = [run("•  ", bold=True, color=BLUE)]
    if bold_prefix and text.startswith(bold_prefix):
        parts.append(run(bold_prefix, bold=True, color=NAVY))
        parts.append(run(text[len(bold_prefix):]))
    else:
        parts.append(run(text))
    return p(parts, left=left, hanging=240, after=70, line=330, keep_lines=True)


def numbered(number: int, text: str) -> str:
    return p(
        [run(f"{number}.  ", bold=True, color=BLUE), run(text)],
        left=420,
        hanging=300,
        after=80,
        line=330,
        keep_lines=True,
    )


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def cell(text_parts: str | list[str], width: int, *, fill: str = WHITE, header: bool = False,
         align: str = "left") -> str:
    if isinstance(text_parts, str):
        text_parts = [run(text_parts, bold=header, color=WHITE if header else TEXT, size=18)]
    tcpr = (
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        '<w:vAlign w:val="center"/>'
        f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
        '<w:tcMar><w:top w:w="90" w:type="dxa"/><w:bottom w:w="90" w:type="dxa"/>'
        '<w:start w:w="110" w:type="dxa"/><w:end w:w="110" w:type="dxa"/></w:tcMar>'
        '<w:tcBorders>'
        f'<w:top w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:left w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:bottom w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:right w:val="single" w:sz="5" w:color="{LINE}"/>'
        '</w:tcBorders>'
    )
    paragraph = p(text_parts, align=align, after=0, line=290, keep_lines=True)
    return f'<w:tc><w:tcPr>{tcpr}</w:tcPr>{paragraph}</w:tc>'


def table(headers: list[str], rows: list[list[str | list[str]]], widths: list[int]) -> str:
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    header_cells = "".join(cell(value, width, fill=NAVY, header=True, align="center")
                           for value, width in zip(headers, widths))
    body_rows = []
    for idx, row in enumerate(rows):
        fill = LIGHT if idx % 2 else WHITE
        cells = "".join(cell(value, width, fill=fill, align="left" if j == 0 else "center")
                        for j, (value, width) in enumerate(zip(row, widths)))
        body_rows.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{cells}</w:tr>')
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblLayout w:type="fixed"/>'
        f'<w:tblBorders><w:top w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:left w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:bottom w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:right w:val="single" w:sz="5" w:color="{LINE}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:color="{LINE}"/>'
        f'<w:insideV w:val="single" w:sz="4" w:color="{LINE}"/>'
        '</w:tblBorders>'
        '</w:tblPr>'
        f'<w:tblGrid>{grid}</w:tblGrid>'
        f'<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>{header_cells}</w:tr>'
        f'{"".join(body_rows)}</w:tbl>'
        + p("", after=80)
    )


def callout(text: str, *, label: str = "一句话", fill: str = LIGHT_BLUE, color: str = DARK_BLUE) -> str:
    parts = [run(f"{label}：", bold=True, color=color, size=20), run(text, color=color, size=20)]
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="9026" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
        '</w:tblPr><w:tblGrid><w:gridCol w:w="9026"/></w:tblGrid><w:tr><w:trPr><w:cantSplit/></w:trPr>'
        + cell(parts, 9026, fill=fill, align="left")
        + '</w:tr></w:tbl>' + p("", after=80)
    )


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", data[16:24])


def picture(rel_id: str, image_path: Path, caption: str, doc_pr_id: int, width_inches: float = 6.05) -> str:
    px_w, px_h = png_size(image_path)
    cx = int(width_inches * 914400)
    cy = int(cx * px_h / px_w)
    drawing = f'''
    <w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="{cx}" cy="{cy}"/>
      <wp:effectExtent l="0" t="0" r="0" b="0"/>
      <wp:docPr id="{doc_pr_id}" name="Figure {doc_pr_id}" descr="{x(caption)}"/>
      <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
      <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
        <pic:pic>
          <pic:nvPicPr><pic:cNvPr id="0" name="{x(image_path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>
          <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
          <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
        </pic:pic>
      </a:graphicData></a:graphic>
    </wp:inline></w:drawing></w:r>'''
    return p([drawing], align="center", after=45, keep_lines=True) + p(
        [run(f"图 {doc_pr_id}　{caption}", italic=True, color=MUTED, size=17)],
        style="Caption",
        align="center",
        after=160,
        line=280,
        keep_lines=True,
    )


def build_body() -> tuple[str, list[tuple[str, Path]]]:
    out: list[str] = []
    image_rels: list[tuple[str, Path]] = []

    # Cover: deliberately plain Word text, no text boxes or formula objects.
    out.append(p("", before=1500, after=0))
    out.append(p([run("RESEARCH PROGRESS · PLAIN-LANGUAGE EDITION", bold=True, color=GOLD, size=19)],
                 style="ReportKicker", align="center", after=300))
    out.append(p([
        run("从 MS-CADM 复现", bold=True, color=NAVY, size=54),
        '<w:r><w:br/></w:r>',
        run("到选择性时间干预", bold=True, color=NAVY, size=54),
    ],
                 style="ReportTitle", align="center", after=260, line=700, keep_lines=True))
    out.append(p([run("风电联合场景生成阶段进展——大白话版", color=BLUE, size=30)],
                 style="ReportSubtitle", align="center", after=300))
    out.append(p([run("GEFCom2014 Wind · 10 个区域 · 24 小时 · 每次生成 100 条场景", color=MUTED, size=19)],
                 style="ReportMeta", align="center", after=90))
    out.append(p([run("整理日期：2026-08-28　｜　适合 20–25 分钟组会汇报", color=MUTED, size=18)],
                 style="ReportMeta", align="center", after=360))
    out.append(callout(
        "我们已经确认“时间顺序有用”，但还没有证明“某个时间模块就是最终答案”。下一步先找出时间信息在什么天气、什么去噪阶段最有价值，再决定最终模型。",
        label="全篇先记住",
        fill=LIGHT_BLUE,
        color=DARK_BLUE,
    ))
    out.append(p([run("说明：这是 train/validation 上的阶段性研究汇报，不是最终测试集结论。", italic=True, color=MUTED, size=18)],
                 align="center", before=180, after=0))
    out.append(page_break())

    out.append(heading("先看结论：目前到底做到哪一步了", 1))
    out.append(plain("如果只用最简单的话概括，目前完成了五件事："))
    for idx, text in enumerate([
        "把 MS-CADM 论文从数据、训练、采样到评价完整重做了一遍；程序能跑，但论文核心数值没有重现。",
        "不再只看一个总分，而是把错误拆成 ramp、时间相关、跨区域关系、边界零状态和天气类型。",
        "重新搭了一套无泄漏、三种子、共同随机数的公平实验框架。",
        "确认了正常时间顺序确实能改善 ramp 和滞后关系，但现有时间模块还不够稳定，因此正式判为 No-Go。",
        "完成 Flow 与 joint DDPM 的公平比较：两者各有长处，没有正式胜者。",
    ], 1):
        out.append(numbered(idx, text))
    out.append(callout(
        "研究主线已经从“给模型再加一个模块”变成“先用实验找到时间信息何时有用，再据此设计模块”。",
        label="当前最大的进展",
        fill=GREEN_LIGHT,
        color=GREEN,
    ))
    out.append(heading("证据状态一览", 2))
    out.append(table(
        ["阶段", "现在的状态", "该怎么理解"],
        [
            ["MS-CADM 复现", "完成", "方法链重建成功；核心数值和 coverage 未复现"],
            ["跨模型诊断", "完成（探索性）", "用于定位问题，不能当最终模型胜负"],
            ["R0 三种子稳定性", "正式通过", "公平比较的地基已经可用"],
            ["时间机制 v3.2", "正式 No-Go", "时间信号成立，但候选稳定性/非劣门失败"],
            ["Flow vs DDPM", "正式无胜者", "不是平手；是各项指标存在权衡"],
            ["新创新路线", "准备冻结协议", "先做 Temporal Utility Curve，再决定完整模型"],
        ],
        [2400, 1900, 4726],
    ))

    out.append(heading("一、这项研究到底想解决什么问题", 1))
    out.append(heading("1. 点预测和场景预测不是一回事", 2))
    out.append(plain(
        "普通风电预测只给一条曲线，好比天气预报只说“明天平均 20℃”。场景预测则要给出很多条可能的完整轨迹：可能一直稳定，也可能上午很低、下午突然升高。它们的平均值可以一样，但对备用容量、机组启停和风险判断的影响完全不同。"
    ))
    out.append(callout(
        "我们的模型不是回答“明天最可能发多少电”，而是回答“明天可能出现哪些 24 小时轨迹，以及这组轨迹是否像真实世界”。",
        label="任务",
    ))
    out.append(table(
        ["项目", "大白话解释", "本研究中的形式"],
        [
            ["输入", "明天的天气预报和区域信息", "10 区域、24 小时 NWP"],
            ["输出", "一组可能发生的完整风电走势", "每个预测日 100 条联合轨迹"],
            ["重点", "不只要均值准，还要变化方式和不确定性合理", "level、ramp、joint、coverage、atom"],
        ],
        [1600, 3900, 3526],
    ))
    out.append(heading("2. 为什么要联合生成十个区域", 2))
    out.append(plain(
        "电网关心的不只是某一个区域。多个区域可能同时升、同时降，也可能一个区域的变化过一小时影响另一个区域。若每个区域完全独立生成，即使单区曲线不错，同一场景编号里的十个区域也可能拼不成一个物理上合理的整体。"
    ))
    out.append(bullet("联合成员语义：第 17 条场景应当同时代表十个区域在同一种可能天气演化下的结果。", bold_prefix="联合成员语义："))
    out.append(bullet("时间语义：一条 24 小时轨迹应当有合理的持续、反弹和爬坡，而不是每小时各猜各的。", bold_prefix="时间语义："))
    out.append(bullet("边界语义：停机或极端状态可能产生精确 0/1，这些概率质量不能全靠连续值裁剪。", bold_prefix="边界语义："))

    out.append(heading("二、为什么从 MS-CADM 开始", 1))
    out.append(plain(
        "MS-CADM 是一篇直接面向风电场景生成的条件扩散论文。它用多尺度天气编码、Transformer 和扩散去噪生成 24 小时场景。它很适合作为起点，因为任务、数据和我们后续想研究的条件生成问题一致。"
    ))
    out.append(heading("1. 我们复现了哪些东西", 2))
    for text in [
        "数据整理：GEFCom2014 的 10 个区域、731 个完整日历日和逐小时 NWP 特征。",
        "模型链路：多尺度条件编码、AdaLN Transformer、噪声与方差预测。",
        "训练链路：26,000 次更新、条件遮蔽、扩散损失和 checkpoint。",
        "生成链路：250 步 ancestral 采样和多种 DDIM 步数对照。",
        "评价链路：MAE、RMSE、CRPS、QS、ES、VS、coverage 和区间宽度。",
    ]:
        out.append(bullet(text))
    out.append(heading("2. 程序能运行，不等于论文结论能重现", 2))
    out.append(plain(
        "本地模型完整训练后，论文最重要的 CRPS 没有对上：论文是 0.0873，本地是 0.1074，差约 23%。更严重的是，名义 90% 区间的实际覆盖率只有 35.95%，说明 100 条场景挤得太紧，很多真实结果落在场景范围之外。"
    ))
    rid = "rId20"
    image_rels.append((rid, IMAGES[0][0]))
    out.append(picture(rid, IMAGES[0][0], IMAGES[0][1], 1))
    out.append(callout(
        "训练 loss 下降只说明网络越来越会完成训练题，不代表它生成的概率分布足够宽、足够准。",
        label="第一个教训",
        fill=AMBER_LIGHT,
        color=AMBER,
    ))
    out.append(heading("3. 为什么不能简单说“论文错了”", 2))
    out.append(plain(
        "论文没有公开完整代码，也缺少网络宽度、深度、噪声日程、损失权重、mask 粒度和 checkpoint 选择等关键细节。因此更准确的结论是：我们按照论文原理和常见扩散实现做出的忠实版本，没有复现论文 headline；不能把本地版本当成作者私有实现的逐行复制。"
    ))

    out.append(heading("三、为什么后来不再只盯着 CRPS 总分", 1))
    out.append(plain(
        "一个模型可能把每天平均功率预测得不错，却把小时之间的变化方式弄错。只看 level CRPS，就像只看一趟车最终到了没到，却不看路上是否一直急刹、急加速。于是我们把问题拆成四类。"
    ))
    out.append(table(
        ["检查项", "大白话含义", "如果做不好会怎样"],
        [
            ["Ramp", "相邻小时是升还是降、变化多大", "场景过抖或漏掉真实爬坡"],
            ["Cross-lag", "不同区域、不同小时之间是否配合得对", "同时关系或前后关系失真"],
            ["Atom duration", "精确零状态出现几次、持续多久", "停机状态太少、太碎或持续错误"],
            ["NWP regime", "错误是否集中在某种天气变化下", "平均分掩盖动态天气失败"],
        ],
        [1900, 3500, 3626],
    ))
    out.append(heading("1. 最重要的发现：平均水平接近，但轨迹更抖", 2))
    out.append(plain(
        "在旧 mixed-measure 联合模型和逐区 DDPM 的配对诊断中，level CRPS 几乎打平；但联合模型的普通 ramp CRPS 差约 5.19%，总变差 CRPS 差约 80.33%，最大跳变 CRPS 差约 57.13%。这说明问题主要出在“怎么走”，而不是“总体在哪”。"
    ))
    rid = "rId21"
    image_rels.append((rid, IMAGES[1][0]))
    out.append(picture(rid, IMAGES[1][0], IMAGES[1][1], 2))
    out.append(heading("2. 空间关系不错，时间前后关系较差", 2))
    out.append(plain(
        "旧联合模型知道“十个区域此刻大致应该一起怎么动”，但不太知道“这一步之后应该继续、反弹还是衰减”。例如同一区域 lag-1 相关误差，旧联合模型是 0.3002，逐区 DDPM 是 0.0149。"
    ))
    rid = "rId22"
    image_rels.append((rid, IMAGES[2][0]))
    out.append(picture(rid, IMAGES[2][0], IMAGES[2][1], 3))
    out.append(callout(
        "联合空间结构不能丢，但时间动态必须直接修。继续只优化平均功率或只堆空间图模块，不会自动解决 lag-1。",
        label="第二个教训",
    ))
    out.append(heading("3. 天气越在变化，问题越明显", 2))
    out.append(plain(
        "平静天气下，旧联合模型和 DDPM 的 ramp 差距约 0.00186；快速变速、明显转向、空间差异大时，差距扩大到约 0.00372、0.00360 和 0.00355。由此产生一个很自然的猜想：时间模块可能不是每天都需要，而是在天气正在变化时更有价值。"
    ))
    rid = "rId23"
    image_rels.append((rid, IMAGES[3][0]))
    out.append(picture(rid, IMAGES[3][0], IMAGES[3][1], 4))
    out.append(p([run("证据提醒：", bold=True, color=AMBER), run(
        "这组旧档案用于提出假设，不用于宣布最终模型胜负；后面重新建立了无泄漏的 Architecture-v1。"
    )], shade=AMBER_LIGHT, left=220, first_line=0, after=140))

    out.append(heading("四、我们怎样把实验做得更公平", 1))
    out.append(plain(
        "早期结果存在一个麻烦：换一个训练 seed 时，不只是网络初始化变了，连天气编码器、边界模块、日期顺序和随机噪声也一起变了。最后看到分数波动，却不知道是谁造成的。Architecture-v1 的工作就是把这些混杂逐个锁住。"
    ))
    out.append(heading("1. 数据不串门", 2))
    for text in [
        "已经看过的旧诊断日永久标为 R-SEEN，不再伪装成全新测试。",
        "缺失值不能跨 train、validation、calibration、selection 边界向前填充。",
        "validation 只用于架构开发；selection 和 calibration 仍然封存。",
    ]:
        out.append(bullet(text))
    out.append(heading("2. 模型之间尽量只差一个变量", 2))
    for text in [
        "共同使用 NWP encoder、atom 模块和 exact-boundary 语义。",
        "共同使用 10 区 × 24 小时联合输出、相同成员数和相同评价脚本。",
        "共同使用日期顺序、训练随机路径、验证随机库、atom allocation 和初始噪声。",
        "比较时间机制时，网络参数量、训练预算、早停和采样 NFE 尽量匹配。",
    ]:
        out.append(bullet(text))
    out.append(heading("3. R0 的种子稳定性已经正式过门", 2))
    out.append(table(
        ["指标", "旧 seed spread", "修复后 spread", "变化"],
        [
            ["level CRPS", "0.003645", "0.001331", "下降 63.5%"],
            ["ramp CRPS", "0.001464", "0.000422", "下降 71.1%"],
            ["joint ES 相对 spread", "3.737%", "1.510%", "下降 59.6%"],
        ],
        [2500, 2100, 2100, 2326],
    ))
    out.append(callout(
        "共同随机数不是“让三个模型变一样”，而是像考试统一试卷和考场，只保留网络初始化这个被检查因素。",
        label="常见疑问",
        fill=GREEN_LIGHT,
        color=GREEN,
    ))

    out.append(heading("五、时间机制实验：为什么“有用”仍然判 No-Go", 1))
    out.append(heading("1. v3.0：模型学会了一种不允许的“捷径”", 2))
    out.append(plain(
        "第一版时间随机源可以自由改变均值、相关性和尺度。训练 loss 的确降了，但它通过把 100 条场景挤在一起让任务变容易：coverage90 掉到 0.507，区间宽度只有 0.171，梯度裁剪比例达到 47.2%。"
    ))
    out.append(callout(
        "就像为了提高“猜中率”，把所有答案都写得很接近。训练题好做了，但不确定性被抹掉了，所以正式停止。",
        label="大白话",
        fill=RED_LIGHT,
        color=RED,
    ))
    out.append(heading("2. v3.1：只允许学习相关性，不允许偷偷缩小方差", 2))
    out.append(plain(
        "修正版使用方差保持的 AR 随机源，只学习前后小时的相关系数，不学习新的均值和尺度。这样 source variance 保持在 1 附近，避免重演场景塌缩。单 seed 小试中，feature recurrence 和 source recurrence 都改善了 ramp 和 lagged dependence。"
    ))
    out.append(heading("3. v3.2：时间顺序信号成立，但候选仍未过安全门", 2))
    out.append(table(
        ["候选", "ramp 改善", "lagged VS 改善", "Shuffle 对照", "正式结论"],
        [
            ["T1-feature", "0.002162", "19.9%", "正常顺序明显更好", "No-Go"],
            ["T1-source", "0.002697", "62.2%", "正常顺序明显更好", "No-Go"],
        ],
        [1900, 1600, 1750, 2100, 1676],
    ))
    out.append(plain(
        "两个正常时间顺序模型都优于对应 Shuffle，说明改善不是单纯来自多几个参数或统一平滑。但是 feature recurrence 的 level/joint 非劣和 joint seed spread 没过门；source recurrence 的 level/joint 跨 seed 波动更大，相关系数还几乎全场顶到上限。"
    ))
    out.append(callout(
        "时间信息是真的有用；问题在于现在把它全天气、全阶段都开着，收益和副作用绑在了一起。",
        label="最准确的解释",
    ))

    out.append(heading("六、Flow 和 Diffusion 到底谁更好", 1))
    out.append(heading("1. 先修复 diffusion 的采样故障", 2))
    out.append(plain(
        "最初 D0 使用 epsilon-prediction。训练数值看起来正常，真正采样时却把 89.28% 的内部坐标打到精确 0/1，连续潜变量甚至扩大到约正负一万二。这不是“分数略差”，而是输出语义坏了，所以旧 family-v1 被永久冻结为 No-Go。"
    ))
    out.append(plain(
        "family-v1.1 只把训练参数化改成 v-prediction，没有同时加 normalization、Min-SNR 或 clipping。修复后，P0 检查、三种子训练、checkpoint resume 和 sampler contract 全部通过。"
    ))
    out.append(heading("2. 正式比较设置", 2))
    for text in [
        "Flow 和 DDPM 各使用 3 个 best EMA checkpoint。",
        "每个 checkpoint 使用 3 个采样 seed，共形成 18 个共同 validation 场景档案。",
        "每个档案包含 50 日 × 100 members × 10 区 × 24 小时。",
        "双方统一使用 31 次网络前向调用，并共享 truth、mask、atom states 和初始噪声。",
    ]:
        out.append(bullet(text))
    out.append(heading("3. 结果不是“谁全面更强”，而是两种长处", 2))
    out.append(table(
        ["指标", "F0 Flow", "D0-v DDPM", "谁更好 / 怎么理解"],
        [
            ["level CRPS ↓", "0.079661", "0.080484", "Flow 好约 1.03%"],
            ["ramp CRPS ↓", "0.051627", "0.051648", "几乎一样"],
            ["joint ES ↓", "0.111059", "0.112640", "Flow 好约 1.40%"],
            ["lagged VS ↓", "0.017720", "0.017189", "DDPM 好约 3.00%"],
            ["coverage90", "0.851658", "0.838193", "Flow 高 1.35 个百分点"],
            ["width90 ↓", "0.378115", "0.368512", "DDPM 更窄，但也更欠覆盖"],
        ],
        [2100, 1700, 1800, 3426],
    ))
    out.append(callout(
        "正式 winner 仍然为空。Flow 的平均/联合分布和 coverage 更好；DDPM 的 lagged dependence 更好。两者都没有满足预注册的全面支配条件。",
        label="正式结论",
        fill=AMBER_LIGHT,
        color=AMBER,
    ))
    out.append(heading("4. 那为什么后续还是用 diffusion 做研究载体", 2))
    out.append(plain(
        "不是因为 DDPM 已经赢了，而是因为新问题正好涉及 diffusion 的“去噪阶段”。Diffusion 每一步都有明确的噪声强弱和 log-SNR；我们可以研究时间信息到底在哪些阶段有用。Flow 仍保留为外部参考基线，不能从报告中消失。"
    ))

    out.append(heading("七、现在准备押注的创新点是什么", 1))
    out.append(heading("1. 不再简单说“给 diffusion 加一个 SSM”", 2))
    out.append(plain(
        "已有文献已经有人把 TCN、SSM、MoE、不同噪声阶段专家等组件放进 diffusion。单独说“我们也加了 SSM”很容易撞车。更有研究价值的问题是：时间信息不是在所有天气、所有去噪阶段都同样有用。"
    ))
    out.append(heading("2. 两个开关的直观解释", 2))
    out.append(table(
        ["部件", "它问的问题", "大白话比喻"],
        [
            ["稳定的 v-base", "不加新时间模块时，基本分布能否站稳", "汽车底盘"],
            ["天气动态开关 g_met", "今天的风正在明显变速、转向或分区变化吗", "今天需不需要辅助驾驶"],
            ["SNR 阶段开关 g_snr", "当前去噪阶段还能不能看清时间结构", "现在是不是适合接管方向盘"],
            ["有界时间残差", "只做多大幅度的补充修正", "限制辅助转向幅度，不能抢走底盘"],
            ["Atom 模块", "哪些位置必须精确等于 0/1", "先决定离散状态，再生成内部连续值"],
        ],
        [1900, 3850, 3276],
    ))
    out.append(callout(
        "基础去噪结果 + 天气开关 × SNR 开关 × 有上限的时间修正。这里全部是普通文本，不使用 Word 公式对象。",
        label="模型一句话",
        fill=GREEN_LIGHT,
        color=GREEN,
    ))
    out.append(heading("3. 创新贡献应该分主次", 2))
    out.append(numbered(1, "主创新：发现并利用“天气状态 × 去噪阶段”的时间信息效用差异。"))
    out.append(numbered(2, "稳定基础：joint v-diffusion、train-only normalization 和 Min-SNR 只作为工程基础，不声称首次提出。"))
    out.append(numbered(3, "系统支撑：把精确边界 atom 与连续内部扩散分开，但不声称 mixed distribution 是我们发明的。"))
    out.append(heading("4. 当前 atom 还存在一个诚实的限制", 2))
    out.append(plain(
        "当前 atom head 给每个区域、每个小时输出三分类概率，再用 shared-priority allocation 让场景成员具有一些共同性。这不是完整学习出的 10 区 × 24 小时联合离散过程；训练用真实 state，推理用预测/分配 state，也存在配置分布差异。"
    ))
    out.append(plain(
        "但 Temporal Utility Probe 中不能同时修改 atom 和时间模块，否则即使分数变好也不知道是谁造成的。因此 probe 阶段先冻结 atom，Atom-v2 另开一条诊断路线。"
    ))

    out.append(heading("八、下一步具体要做什么", 1))
    out.append(heading("1. 先把 diffusion 基线稳定住", 2))
    out.append(plain(
        "下一协议暂定为 family-v1.2。D1 使用只由 train 估计的连续潜变量 normalization；D2 再测试适配 v-prediction 的 Min-SNR weighting。它们不是自动叠加的升级包：每次只改一项，没有通过预注册优势就退回更简单的 D0-v。"
    ))
    out.append(heading("2. Temporal Utility Probe：先画曲线，不直接写完整双门控", 2))
    for idx, text in enumerate([
        "冻结 diffusion base，不再更新它的参数。",
        "把训练 timestep 映射为 8 个 log-SNR 区间。",
        "每个区间单独训练一个很小、零初始化、有幅度上限的时间 residual adapter。",
        "每个区间都做 Chronological 与固定 Shuffle 对照。",
        "使用 3 个训练 seed、共同噪声、共同 atom allocation 和相同 validation 日。",
        "再按 train-only NWP dynamicity 检查 stable 与 dynamic 天气上的收益是否不同。",
    ], 1):
        out.append(numbered(idx, text))
    out.append(callout(
        "正式 retained 矩阵是 8 个 SNR 区间 × 2 种时间顺序 × 3 seeds = 48 个轻量 adapter run。它比直接训练一个复杂 M3 更能回答科学问题。",
        label="实验规模",
    ))
    out.append(heading("3. 什么结果才允许继续做双门控", 2))
    out.append(table(
        ["检查", "预期门槛", "如果没过怎么办"],
        [
            ["ramp", "改善至少 0.001，配对区间支持", "不保留时间机制"],
            ["lagged VS", "相对改善至少 5%", "不能只凭 ramp 宣布成功"],
            ["level / joint", "恶化不超过 0.0015 / 2%", "防止顾此失彼"],
            ["coverage / width", "差不超过 0.02 / 增幅不超 10%", "防止缩窄或无差别扩宽"],
            ["SNR 结构", "至少两个相邻区间形成稳定峰", "曲线平坦则取消 SNR gate"],
            ["天气交互", "dynamic 收益高于 stable", "不成立则取消天气 gate"],
            ["Shuffle", "正常顺序必须更好", "否则说明并非真实时间信息"],
        ],
        [1900, 3900, 3226],
    ))
    out.append(heading("4. 最后有三种可能", 2))
    out.append(numbered(1, "SNR 峰值和天气交互都成立：实现完整 M3 双门控。"))
    out.append(numbered(2, "只成立一个：只保留相应单门，不为了故事强行相乘。"))
    out.append(numbered(3, "曲线平坦或 Shuffle 同样好：停止门控路线，保留更简单基线。"))

    out.append(heading("九、现在可以说什么，不能说什么", 1))
    out.append(heading("可以说", 2))
    for text in [
        "MS-CADM 方法和实验链已完整工程重构，但核心数值和可靠性未复现。",
        "旧联合模型的主要缺口集中在 ramp、lag-1 和动态天气条件。",
        "正常时间顺序显著优于 Shuffle，说明时间信息真实存在。",
        "F0 与 D0-v 都能稳定生成联合场景，但在边际/联合质量和 lagged dependence 上存在权衡。",
        "Temporal Utility Curve 是当前最有信息量、最可证伪的下一实验。",
    ]:
        out.append(bullet(text))
    out.append(heading("不能说", 2))
    for text in [
        "不能说 diffusion 已经战胜 Flow；正式 winner 为空。",
        "不能说 T1-feature 或 T1-source 是最终模型；它们正式 No-Go。",
        "不能把 validation 改善当成最终测试或外部泛化。",
        "不能说当前 atom 模块已经学习完整联合状态过程。",
        "不能声称首次使用 diffusion、SSM、SNR gate 或 mixed distribution。",
    ]:
        out.append(bullet(text))

    out.append(heading("十、组会上可以怎样收尾", 1))
    out.append(callout(
        "我们从一个没有复现核心数值、且严重欠覆盖的 MS-CADM 重实现出发，通过跨模型诊断确认问题主要在轨迹动态；再通过三种子和 Shuffle 实验证明时间顺序确实有价值，但当前统一时间机制会损害稳定性。Flow 与 v-DDPM 没有胜者，因此下一步不是拍脑袋选架构，而是先测量天气状态 × SNR 的时间效用曲线。",
        label="60 秒总结",
    ))
    out.append(heading("希望组内讨论的三个问题", 2))
    out.append(numbered(1, "8-bin × Shuffle × 3-seed 的 Temporal Utility Probe 是否足以支撑方法设计？"))
    out.append(numbered(2, "论文主贡献是否应聚焦 selective temporal intervention，把 mixed-measure 和 v-diffusion 降为基础？"))
    out.append(numbered(3, "Atom-v2 应放在同一篇论文中作为支撑，还是拆成下一项独立工作？"))

    out.append(heading("附录 A：常见指标的大白话解释", 1))
    out.append(table(
        ["指标", "它在问什么", "怎么看"],
        [
            ["level CRPS", "每个小时的概率分布离真实值多远", "越低越好"],
            ["ramp CRPS", "相邻小时变化量的概率分布准不准", "越低越好"],
            ["joint ES", "把 10 区 × 24 小时当整体后，场景像不像真实联合轨迹", "越低越好"],
            ["lagged VS", "不同小时、不同区域之间的差异关系像不像真的", "越低越好"],
            ["coverage90", "真实值落在 90% 场景区间内的比例", "应接近 0.90"],
            ["width90", "90% 场景区间有多宽", "必须和 coverage 一起看"],
            ["atom Brier", "精确 0/1/interior 三种状态概率准不准", "越低越好"],
            ["seed spread", "换网络初始化后结果漂不漂", "越小越稳定"],
        ],
        [1800, 5200, 2026],
    ))

    out.append(heading("附录 B：可复核的仓库材料", 1))
    evidence = [
        "reports/MSCADM_PAPER_REPRODUCTION_REPORT.md —— 原论文方法、复现数值和审计边界",
        "reports/CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md —— ramp、cross-lag、atom、NWP regime 诊断",
        "reports/ARCHITECTURE_V1_R0_STABILITY_V2_3_FORMAL_RESULT.md —— 三种子稳定性正式结果",
        "reports/ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md —— 时间机制与 Shuffle 正式裁决",
        "reports/ARCHITECTURE_V1_FAMILY_V1_1_TRAINING_RESULT.md —— v-pred 修订和 retained 训练",
        "reports/ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md —— Flow vs DDPM 正式比较",
        "reports/CURRENT_RESEARCH_PROGRESS_GROUP_MEETING_STANDALONE.html —— 同主题浏览器图文版",
    ]
    for item in evidence:
        out.append(bullet(item))
    out.append(p([run("文档结束", bold=True, color=GOLD, size=18)], align="center", before=320, after=0))

    return "".join(out), image_rels


def header_xml(text: str = "") -> str:
    body = p([run(text, bold=True, color=MUTED, size=16)], style="Header", align="left", after=0) if text else p("", after=0)
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr xmlns:w="{NS["w"]}" xmlns:r="{NS["r"]}">{body}</w:hdr>'


def footer_xml(blank: bool = False) -> str:
    if blank:
        body = p("", after=0)
    else:
        field = (
            run("风电场景生成阶段进展 · 大白话版　｜　第 ", color=MUTED, size=16)
            + '<w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:color w:val="66737F"/><w:sz w:val="16"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'
            + run(" 页", color=MUTED, size=16)
        )
        body = p([field], style="Footer", align="center", after=0)
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="{NS["w"]}" xmlns:r="{NS["r"]}">{body}</w:ftr>'


def document_xml(body: str) -> str:
    namespaces = " ".join(f'xmlns:{name}="{uri}"' for name, uri in NS.items())
    sect = (
        '<w:sectPr>'
        '<w:headerReference w:type="first" r:id="rId9"/>'
        '<w:footerReference w:type="first" r:id="rId10"/>'
        '<w:headerReference w:type="default" r:id="rId11"/>'
        '<w:footerReference w:type="default" r:id="rId12"/>'
        '<w:titlePg/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1250" w:right="1440" w:bottom="1250" w:left="1440" w:header="600" w:footer="600" w:gutter="0"/>'
        '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/>'
        '</w:sectPr>'
    )
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {namespaces}><w:body>{body}{sect}</w:body></w:document>'


def rels_xml(image_rels: list[tuple[str, Path]]) -> str:
    base = [
        ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "styles.xml"),
        ("rId4", "http://schemas.microsoft.com/office/2007/relationships/stylesWithEffects", "stylesWithEffects.xml"),
        ("rId5", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings", "settings.xml"),
        ("rId6", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings", "webSettings.xml"),
        ("rId7", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable", "fontTable.xml"),
        ("rId8", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
        ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering", "numbering.xml"),
        ("rId9", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header", "header1.xml"),
        ("rId10", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer", "footer1.xml"),
        ("rId11", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header", "header2.xml"),
        ("rId12", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer", "footer2.xml"),
    ]
    relationships = [f'<Relationship Id="{rid}" Type="{rtype}" Target="{target}"/>' for rid, rtype, target in base]
    for index, (rid, _path) in enumerate(image_rels, 1):
        relationships.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/plain_progress_{index}.png"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + '</Relationships>'
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>从 MS-CADM 复现到选择性时间干预：阶段进展大白话版</dc:title>
 <dc:subject>风电联合场景生成阶段进展</dc:subject><dc:creator>multi_scaleCADM</dc:creator>
 <cp:keywords>wind power; scenario generation; diffusion; rectified flow; temporal utility</cp:keywords>
 <dc:description>不使用 Word 公式对象的 WPS 兼容大白话版阶段汇报。</dc:description>
 <cp:lastModifiedBy>multi_scaleCADM</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''


def app_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
 <Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>
 <Company>multi_scaleCADM</Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc>
 <HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>
</Properties>'''


def build() -> tuple[Path, str]:
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    for path, _caption in IMAGES:
        if not path.exists():
            raise FileNotFoundError(path)

    body, image_rels = build_body()
    document = document_xml(body)
    rels = rels_xml(image_rels)

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
        zout.writestr("docProps/app.xml", app_xml().encode("utf-8"))
        zout.writestr("word/header1.xml", header_xml().encode("utf-8"))
        zout.writestr("word/footer1.xml", footer_xml(blank=True).encode("utf-8"))
        zout.writestr("word/header2.xml", header_xml("风电联合场景生成阶段进展 · 大白话版").encode("utf-8"))
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
