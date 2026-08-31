#!/usr/bin/env python3
"""Build a self-contained, offline HTML edition of the research roadmap."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import re
from pathlib import Path

from markdown_it import MarkdownIt


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FUTURE_RESEARCH_ROADMAP.md"
OUTPUT = HERE / "FUTURE_RESEARCH_ROADMAP_STANDALONE.html"


def slugify(value: str, used: set[str]) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.lower(), flags=re.UNICODE)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-") or "section"
    candidate = cleaned
    counter = 2
    while candidate in used:
        candidate = f"{cleaned}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def embed_images(rendered: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        prefix, source, suffix = match.groups()
        if source.startswith(("data:", "http://", "https://")):
            return match.group(0)
        path = (HERE / html.unescape(source)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image referenced by Markdown does not exist: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        count += 1
        return f'{prefix}data:{mime};base64,{payload}{suffix}'

    pattern = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', re.IGNORECASE)
    return pattern.sub(replace, rendered), count


def render_markdown(markdown_text: str) -> tuple[str, list[dict[str, str]]]:
    parser = MarkdownIt("commonmark", {"html": True, "typographer": True})
    parser.enable("table")
    parser.enable("strikethrough")
    tokens = parser.parse(markdown_text)
    used: set[str] = set()
    toc: list[dict[str, str]] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        level = int(token.tag[1])
        title = tokens[index + 1].content.strip()
        anchor = slugify(title, used)
        token.attrSet("id", anchor)
        if level in (2, 3):
            toc.append({"level": str(level), "title": title, "anchor": anchor})

    rendered = parser.renderer.render(tokens, parser.options, {})
    rendered = re.sub(r"<table>", '<div class="table-wrap"><table>', rendered)
    rendered = re.sub(r"</table>", "</table></div>", rendered)
    rendered = re.sub(
        r"<li>\[ \] ",
        '<li class="task"><input type="checkbox" disabled aria-label="未完成"> ',
        rendered,
    )
    rendered = re.sub(
        r"<li>\[[xX]\] ",
        '<li class="task"><input type="checkbox" checked disabled aria-label="已完成"> ',
        rendered,
    )
    return rendered, toc


def build_toc(items: list[dict[str, str]]) -> str:
    links = []
    for item in items:
        level_class = "toc-sub" if item["level"] == "3" else "toc-main"
        links.append(
            f'<a class="toc-link {level_class}" href="#{html.escape(item["anchor"])}">'
            f'{html.escape(item["title"])}</a>'
        )
    return "\n".join(links)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    markdown_text = SOURCE.read_text(encoding="utf-8")
    body, toc = render_markdown(markdown_text)
    body, embedded_count = embed_images(body)
    toc_html = build_toc(toc)
    source_sha = hashlib.sha256(markdown_text.encode("utf-8")).hexdigest()
    section_count = sum(item["level"] == "2" for item in toc)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="MS-CADM 后续研究路线决策、实验协议与论文蓝图">
<title>MS-CADM 后续研究路线决策与实验蓝图</title>
<style>
:root {{
  --navy: #17365d; --blue: #286aa3; --blue-soft: #eaf3fa;
  --gold: #bd8424; --green: #217a48; --green-soft: #eaf6ef;
  --red: #b8384b; --red-soft: #fceef0; --amber: #a96600;
  --ink: #24313d; --muted: #647484; --line: #d7e0e8;
  --surface: #fff; --surface-2: #f6f8fa; --canvas: #eaf0f5;
  --code: #132b45; --shadow: 0 12px 38px rgba(24,52,83,.12);
  --sidebar: 324px; --reader: 1040px;
}}
html.dark {{
  --navy: #91c8f4; --blue: #80bce9; --blue-soft: #20384b;
  --gold: #e9bd69; --green: #72cc94; --green-soft: #1e3b2c;
  --red: #ff8997; --red-soft: #47262e; --amber: #f1b75d;
  --ink: #e7edf3; --muted: #a9b6c2; --line: #3a4a59;
  --surface: #18232e; --surface-2: #202e3b; --canvas: #101820;
  --code: #0e1b28; --shadow: 0 12px 38px rgba(0,0,0,.28);
}}
* {{ box-sizing: border-box; }}
html {{ max-width: 100%; overflow-x: hidden; scroll-behavior: smooth; background: var(--canvas); color: var(--ink); }}
body {{ max-width: 100%; margin: 0; overflow-x: hidden; font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; font-size: 16px; line-height: 1.78; }}
a {{ color: var(--blue); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.progress {{ position: fixed; inset: 0 0 auto; height: 4px; z-index: 120; background: transparent; }}
.progress > span {{ display: block; width: 0; height: 100%; background: linear-gradient(90deg,var(--blue),var(--gold)); }}
.topbar {{ position: fixed; z-index: 110; top: 4px; left: 0; right: 0; height: 54px; display: flex; align-items: center; gap: 12px; padding: 0 18px; color: white; background: rgba(23,54,93,.97); box-shadow: 0 2px 12px rgba(0,0,0,.18); }}
.brand {{ font-weight: 800; letter-spacing: .02em; white-space: nowrap; color: white; }}
.topbar .spacer {{ flex: 1; }}
.topbar button {{ border: 1px solid rgba(255,255,255,.46); border-radius: 18px; padding: 6px 12px; color: white; background: transparent; cursor: pointer; font: inherit; font-size: 13px; }}
.topbar button:hover {{ background: rgba(255,255,255,.12); }}
.menu-button {{ display: none; }}
.sidebar {{ position: fixed; z-index: 100; top: 58px; bottom: 0; left: 0; width: var(--sidebar); padding: 20px 18px 28px; overflow-y: auto; background: var(--surface); border-right: 1px solid var(--line); }}
.offline {{ margin: 0 0 14px; padding: 11px 12px; border-radius: 8px; background: var(--green-soft); color: var(--green); font-size: 12px; line-height: 1.5; font-weight: 700; }}
.search {{ width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); background: var(--surface-2); font: inherit; font-size: 13px; outline: none; }}
.search:focus {{ border-color: var(--blue); box-shadow: 0 0 0 3px color-mix(in srgb,var(--blue) 18%, transparent); }}
.toc {{ margin-top: 14px; }}
.toc-link {{ display: block; padding: 7px 10px; border-left: 3px solid transparent; border-radius: 0 6px 6px 0; color: var(--muted); font-size: 12.5px; line-height: 1.38; }}
.toc-link:hover {{ color: var(--blue); background: var(--blue-soft); text-decoration: none; }}
.toc-link.active {{ color: var(--blue); background: var(--blue-soft); border-left-color: var(--blue); font-weight: 700; }}
.toc-sub {{ padding-left: 24px; font-size: 11.5px; }}
.toc-link.filtered {{ display: none; }}
.layout {{ min-width: 0; margin-left: var(--sidebar); padding: 88px 40px 90px; }}
.reader {{ width: 100%; min-width: 0; max-width: var(--reader); margin: 0 auto; padding: 0 52px 72px; background: var(--surface); border-radius: 14px; box-shadow: var(--shadow); overflow: hidden; }}
.hero {{ margin: 0 -52px 42px; padding: 64px 52px 44px; color: white; background: #17365d; border-top: 8px solid #bd8424; }}
.hero-kicker {{ color: #f0c46e; font-weight: 800; letter-spacing: .14em; font-size: 12px; }}
.hero h1 {{ margin: 12px 0 10px; color: white; font-size: clamp(30px,5vw,50px); line-height: 1.18; border: 0; }}
.hero p {{ max-width: 840px; margin: 0; color: rgba(255,255,255,.9); font-size: 17px; }}
.hero-stats {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; margin-top: 28px; }}
.stat {{ padding: 13px 15px; border: 1px solid rgba(255,255,255,.24); border-radius: 10px; background: rgba(255,255,255,.09); }}
.stat strong {{ display: block; color: white; font-size: 21px; }}
.stat span {{ color: rgba(255,255,255,.75); font-size: 12px; }}
.browser-note {{ margin: -20px 0 30px; padding: 12px 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-2); color: var(--muted); font-size: 13px; }}
article > h1:first-of-type, article > h1:first-of-type + blockquote, article > h1:first-of-type + blockquote + p {{ display: none; }}
h1,h2,h3,h4 {{ color: var(--navy); line-height: 1.35; scroll-margin-top: 76px; }}
h2 {{ margin: 66px -52px 28px; padding: 24px 52px 14px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: linear-gradient(90deg,var(--surface-2),transparent); font-size: 28px; }}
h2::before {{ content: "RESEARCH ROADMAP"; display: block; margin-bottom: 5px; color: var(--gold); font-size: 10px; letter-spacing: .16em; }}
h3 {{ margin: 34px 0 14px; font-size: 21px; }}
h4 {{ margin: 25px 0 10px; font-size: 17px; }}
p {{ margin: 0 0 15px; overflow-wrap: anywhere; }}
strong {{ color: var(--navy); }}
blockquote {{ margin: 22px 0; padding: 16px 20px; border-left: 5px solid var(--blue); border-radius: 0 8px 8px 0; background: var(--blue-soft); color: var(--navy); }}
blockquote p:last-child {{ margin-bottom: 0; }}
ul,ol {{ margin: 8px 0 20px; padding-left: 28px; }}
li {{ margin: 5px 0; }}
li.task {{ list-style: none; margin-left: -24px; }}
li.task input {{ margin-right: 8px; accent-color: var(--blue); }}
hr {{ margin: 52px 0; border: 0; border-top: 1px solid var(--line); }}
code {{ padding: 2px 6px; border-radius: 5px; color: var(--blue); background: var(--blue-soft); font-family: Consolas,"SFMono-Regular",monospace; font-size: .91em; overflow-wrap: anywhere; }}
pre {{ position: relative; margin: 20px 0 25px; padding: 18px 20px; overflow-x: auto; border-radius: 10px; color: #eff7ff; background: var(--code); box-shadow: inset 0 1px 0 rgba(255,255,255,.07); white-space: pre-wrap; overflow-wrap: anywhere; }}
pre::before {{ content: "PLAIN TEXT · UTF-8"; display: block; margin-bottom: 9px; color: #83b7df; font-family: Arial,sans-serif; font-size: 9px; letter-spacing: .13em; }}
pre code {{ padding: 0; color: inherit; background: transparent; font-size: 13px; line-height: 1.58; }}
.table-wrap {{ width: 100%; margin: 19px 0 28px; overflow-x: auto; border: 1px solid var(--line); border-radius: 9px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.45; }}
thead {{ position: sticky; top: 0; z-index: 1; }}
th {{ padding: 11px 12px; color: white; background: var(--navy); text-align: left; white-space: nowrap; }}
html.dark th {{ color: #102030; background: #91c8f4; }}
td {{ min-width: 105px; padding: 10px 12px; border-top: 1px solid var(--line); vertical-align: top; }}
tbody tr:nth-child(even) {{ background: var(--surface-2); }}
tbody tr:hover {{ background: var(--blue-soft); }}
img {{ display: block; max-width: 100%; max-height: 760px; margin: 28px auto; border: 1px solid var(--line); border-radius: 9px; box-shadow: 0 8px 28px rgba(23,54,93,.10); object-fit: contain; }}
.source-meta {{ margin: 60px 0 0; padding: 15px 18px; border-radius: 8px; background: var(--surface-2); color: var(--muted); font-size: 12px; word-break: break-all; }}
.back-top {{ position: fixed; right: 22px; bottom: 22px; z-index: 90; display: none; width: 42px; height: 42px; border: 0; border-radius: 50%; color: white; background: var(--blue); box-shadow: var(--shadow); cursor: pointer; font-size: 18px; }}
.back-top.show {{ display: block; }}
.print-only {{ display: none; }}
@media (max-width: 1020px) {{
  .menu-button {{ display: inline-block; }}
  .sidebar {{ transform: translateX(-102%); transition: transform .2s ease; box-shadow: var(--shadow); }}
  .sidebar.open {{ transform: translateX(0); }}
  .layout {{ margin-left: 0; padding: 78px 18px 60px; }}
  .reader {{ padding: 0 30px 60px; }}
  .hero {{ margin-left: -30px; margin-right: -30px; padding-left: 30px; padding-right: 30px; }}
  h2 {{ margin-left: -30px; margin-right: -30px; padding-left: 30px; padding-right: 30px; }}
}}
@media (max-width: 640px) {{
  body {{ font-size: 15px; }}
  .topbar {{ gap: 6px; padding: 0 8px; }}
  .topbar button {{ padding: 5px 9px; }}
  .topbar #print-button {{ display: none; }}
  .brand {{ flex: 1; min-width: 0; max-width: none; overflow: hidden; text-overflow: ellipsis; font-size: 14px; }}
  .layout {{ padding-left: 0; padding-right: 0; }}
  .reader {{ border-radius: 0; padding: 0 18px 50px; }}
  .hero {{ margin-left: -18px; margin-right: -18px; padding: 48px 18px 32px; }}
  .hero-stats {{ grid-template-columns: 1fr; }}
  h2 {{ margin-left: -18px; margin-right: -18px; padding-left: 18px; padding-right: 18px; font-size: 24px; }}
  .toc-sub {{ display: none; }}
}}
@media print {{
  @page {{ size: A4; margin: 15mm 14mm 17mm; }}
  html,body {{ background: white !important; color: #222 !important; font-size: 9.5pt; }}
  .topbar,.sidebar,.progress,.back-top,.browser-note {{ display: none !important; }}
  .layout {{ margin: 0; padding: 0; }}
  .reader {{ max-width: none; margin: 0; padding: 0; box-shadow: none; overflow: visible; }}
  .hero {{ min-height: 245mm; margin: 0 0 15mm; padding: 70mm 15mm 20mm; page-break-after: always; print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  h2 {{ margin: 12mm 0 5mm; padding: 4mm 0 2mm; page-break-before: always; background: none; }}
  h3,h4 {{ break-after: avoid; }}
  p,li {{ orphans: 3; widows: 3; }}
  pre,blockquote,img,.table-wrap {{ break-inside: avoid; }}
  .table-wrap {{ overflow: visible; }}
  table {{ font-size: 7pt; }}
  th {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  a {{ color: inherit; }}
  .source-meta {{ break-before: page; }}
  .print-only {{ display: block; }}
}}
</style>
</head>
<body>
<div class="progress" aria-hidden="true"><span id="progress-fill"></span></div>
<header class="topbar">
  <button class="menu-button" id="menu-button" type="button" aria-label="打开目录">目录</button>
  <a class="brand" href="#top">MS-CADM 后续路线</a>
  <span class="spacer"></span>
  <button id="font-button" type="button" aria-label="调整字号">字号</button>
  <button id="theme-button" type="button" aria-label="切换深浅色">深色</button>
  <button id="print-button" type="button" aria-label="打印文档">打印</button>
</header>
<aside class="sidebar" id="sidebar" aria-label="文档目录">
  <div class="offline">离线单文件 · 图片已内嵌<br>公式为 UTF-8 纯文本，不依赖 MathJax</div>
  <input class="search" id="toc-search" type="search" placeholder="筛选目录…" aria-label="筛选目录">
  <nav class="toc">{toc_html}</nav>
</aside>
<main class="layout" id="top">
  <div class="reader">
    <section class="hero">
      <div class="hero-kicker">RESEARCH ROADMAP · DECISION DOCUMENT</div>
      <h1>MS-CADM 后续研究路线<br>决策与实验蓝图</h1>
      <p>从已确认的仓库证据出发，明确下一步研究问题、模型结构、消融、统计协议、停止门槛与投稿边界。</p>
      <div class="hero-stats">
        <div class="stat"><strong>{section_count}</strong><span>个主章节</span></div>
        <div class="stat"><strong>3</strong><span>条分级主路线</span></div>
        <div class="stat"><strong>Offline</strong><span>无外部渲染依赖</span></div>
      </div>
    </section>
    <div class="browser-note">使用左侧目录跳转；可切换深色和字号。按 <strong>Ctrl + P</strong> 可打印或另存 PDF。文内本地资料链接在仓库原位置打开。</div>
    <article>{body}</article>
    <div class="source-meta">
      生成源：FUTURE_RESEARCH_ROADMAP.md · SHA256 {source_sha}<br>
      内嵌图片：{embedded_count} · 生成器：build_future_research_roadmap_html.py
    </div>
  </div>
</main>
<button class="back-top" id="back-top" type="button" aria-label="返回顶部">↑</button>
<script>
const root = document.documentElement;
const themeButton = document.getElementById('theme-button');
const fontButton = document.getElementById('font-button');
const sidebar = document.getElementById('sidebar');
const menuButton = document.getElementById('menu-button');
const backTop = document.getElementById('back-top');
const progress = document.getElementById('progress-fill');
const storage = {{
  get(key, fallback) {{ try {{ return localStorage.getItem(key) ?? fallback; }} catch (_) {{ return fallback; }} }},
  set(key, value) {{ try {{ localStorage.setItem(key, value); }} catch (_) {{ /* file:// privacy mode */ }} }}
}};
const savedTheme = storage.get('roadmap-theme', 'light');
if (savedTheme === 'dark') root.classList.add('dark');
themeButton.textContent = root.classList.contains('dark') ? '浅色' : '深色';
themeButton.addEventListener('click', () => {{
  root.classList.toggle('dark');
  storage.set('roadmap-theme', root.classList.contains('dark') ? 'dark' : 'light');
  themeButton.textContent = root.classList.contains('dark') ? '浅色' : '深色';
}});
const fontSizes = ['15px','16px','18px'];
let fontIndex = Number(storage.get('roadmap-font', '1'));
document.body.style.fontSize = fontSizes[fontIndex];
fontButton.addEventListener('click', () => {{
  fontIndex = (fontIndex + 1) % fontSizes.length;
  document.body.style.fontSize = fontSizes[fontIndex];
  storage.set('roadmap-font', String(fontIndex));
}});
document.getElementById('print-button').addEventListener('click', () => window.print());
menuButton.addEventListener('click', () => sidebar.classList.toggle('open'));
document.querySelectorAll('.toc-link').forEach(link => link.addEventListener('click', () => sidebar.classList.remove('open')));
document.getElementById('toc-search').addEventListener('input', event => {{
  const query = event.target.value.trim().toLowerCase();
  document.querySelectorAll('.toc-link').forEach(link => {{
    link.classList.toggle('filtered', query && !link.textContent.toLowerCase().includes(query));
  }});
}});
function updateScroll() {{
  const height = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = (height > 0 ? (window.scrollY / height) * 100 : 0) + '%';
  backTop.classList.toggle('show', window.scrollY > 700);
}}
window.addEventListener('scroll', updateScroll, {{passive:true}});
updateScroll();
backTop.addEventListener('click', () => window.scrollTo({{top:0,behavior:'smooth'}}));
const tocLinks = new Map([...document.querySelectorAll('.toc-link')].map(a => [a.getAttribute('href').slice(1), a]));
const observer = new IntersectionObserver(entries => {{
  entries.forEach(entry => {{
    if (!entry.isIntersecting) return;
    document.querySelectorAll('.toc-link.active').forEach(a => a.classList.remove('active'));
    const link = tocLinks.get(entry.target.id);
    if (link) link.classList.add('active');
  }});
}}, {{rootMargin:'-68px 0px -75% 0px',threshold:0}});
document.querySelectorAll('h2[id],h3[id]').forEach(h => observer.observe(h));
</script>
</body>
</html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "output": str(OUTPUT),
                "source_sha256": source_sha,
                "sections": section_count,
                "toc_entries": len(toc),
                "embedded_images": embedded_count,
                "bytes": OUTPUT.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
