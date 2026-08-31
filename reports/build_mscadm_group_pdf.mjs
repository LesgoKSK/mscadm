import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { spawnSync } from "child_process";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const source = path.join(here, "MSCADM_GROUP_MEETING_GUIDE.md");
const htmlOutput = path.join(here, "MSCADM_GROUP_MEETING_GUIDE.html");
const pdfOutput = path.join(here, "MSCADM_GROUP_MEETING_GUIDE.pdf");
const previewOutput = path.join(here, "qa", "MSCADM_GROUP_MEETING_GUIDE_cover.png");
const linuxChrome = process.env.CHROME_PATH || "/usr/bin/google-chrome";
const windowsChrome = process.env.WINDOWS_CHROME_PATH || "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe";

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function inline(value) {
  const placeholders = [];
  let text = escapeHtml(value);
  text = text.replace(/`([^`]+)`/g, (_match, code) => {
    const token = `@@CODE${placeholders.length}@@`;
    placeholders.push(`<code>${code}</code>`);
    return token;
  });
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, label, target) => {
    const resolved = target.startsWith("http") ? target : target;
    return `<a href="${resolved}">${label}</a>`;
  });
  placeholders.forEach((value, index) => {
    text = text.replace(`@@CODE${index}@@`, value);
  });
  return text;
}

function renderMarkdown(raw) {
  const lines = raw.replace(/\r\n/g, "\n").split("\n");
  const startIndex = lines.findIndex((line) => line.trim() === "## 使用方式");
  const output = [];
  let paragraph = [];
  let listType = null;
  let code = null;
  let quote = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      output.push(`<p>${inline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listType) {
      output.push(`</${listType}>`);
      listType = null;
    }
  };
  const flushQuote = () => {
    if (quote.length) {
      output.push(`<blockquote>${quote.map((line) => `<p>${inline(line)}</p>`).join("")}</blockquote>`);
      quote = [];
    }
  };
  const flushAll = () => {
    flushParagraph();
    flushList();
    flushQuote();
  };

  for (let index = Math.max(startIndex, 1); index < lines.length; index += 1) {
    const line = lines[index];

    if (code !== null) {
      if (line.startsWith("```")) {
        output.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = null;
      } else {
        code.push(line);
      }
      continue;
    }
    if (line.startsWith("```")) {
      flushAll();
      code = [];
      continue;
    }
    if (line.startsWith("![")) {
      flushAll();
      const match = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
      if (match) {
        output.push(`<figure><img src="${match[2]}" alt="${escapeHtml(match[1])}"><figcaption>${inline(match[1])}</figcaption></figure>`);
      }
      continue;
    }
    if (line.startsWith("#")) {
      flushAll();
      const match = line.match(/^(#{1,3})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const title = match[2];
        let className = "";
        let id = "";
        if (level === 1) className = "part";
        if (level === 2 && /^第 \d+ 页/.test(title)) {
          className = "talk-page";
          id = `page-${title.match(/^第 (\d+) 页/)[1]}`;
        }
        if (level === 2 && title === "使用方式") className = "usage";
        if (level === 3 && title === "屏幕上放什么") className = "screen-label";
        if (level === 3 && title === "怎么讲") className = "speaker-label";
        if (level === 3 && title === "过渡句") className = "transition-label";
        output.push(`<h${level}${className ? ` class="${className}"` : ""}${id ? ` id="${id}"` : ""}>${inline(title)}</h${level}>`);
      }
      continue;
    }
    if (/^---+$/.test(line.trim())) {
      flushAll();
      output.push("<hr>");
      continue;
    }
    if (line.startsWith("> ")) {
      flushParagraph();
      flushList();
      quote.push(line.slice(2));
      continue;
    }
    if (line.trim().startsWith("|")) {
      flushAll();
      const rows = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        rows.push(lines[index].trim().slice(1, -1).split("|").map((cell) => cell.trim()));
        index += 1;
      }
      index -= 1;
      if (rows.length >= 2) {
        const header = rows[0];
        const body = rows.slice(2);
        output.push("<div class=\"table-wrap\"><table><thead><tr>" + header.map((cell) => `<th>${inline(cell)}</th>`).join("") + "</tr></thead><tbody>");
        for (const row of body) {
          output.push("<tr>" + row.map((cell) => `<td>${inline(cell)}</td>`).join("") + "</tr>");
        }
        output.push("</tbody></table></div>");
      }
      continue;
    }
    const bullet = line.match(/^\s*-\s+(.+)$/);
    const numbered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (bullet || numbered) {
      flushParagraph();
      flushQuote();
      const wanted = bullet ? "ul" : "ol";
      if (listType !== wanted) {
        flushList();
        listType = wanted;
        output.push(`<${listType}>`);
      }
      output.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      continue;
    }
    if (line.trim() === "\\[") {
      flushAll();
      const formula = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "\\]") {
        formula.push(lines[index].trim());
        index += 1;
      }
      output.push(`<div class="formula">${escapeHtml(formula.join(" "))}</div>`);
      continue;
    }
    if (!line.trim()) {
      flushAll();
      continue;
    }
    flushQuote();
    flushList();
    paragraph.push(line.trim());
  }
  flushAll();
  return output.join("\n");
}

const raw = fs.readFileSync(source, "utf8");
const body = renderMarkdown(raw);
const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MS-CADM 原论文复现：组会讲稿</title>
<style>
@page { size: A4 portrait; margin: 15mm 15mm 17mm; }
* { box-sizing: border-box; }
html { color: #263238; background: #fff; font-family: "Microsoft YaHei", "DengXian", sans-serif; }
body { margin: 0; font-size: 10.2pt; line-height: 1.52; }
.cover { height: 264mm; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; page-break-after: always; color: #17365D; position: relative; overflow: hidden; }
.cover::before { content: ""; position: absolute; z-index: 0; width: 180mm; height: 180mm; border: 32mm solid #EAF2F8; border-radius: 50%; top: -100mm; right: -105mm; }
.cover::after { content: ""; position: absolute; z-index: 0; width: 150mm; height: 150mm; border: 1.5mm solid #C59A3D; border-radius: 50%; bottom: -112mm; left: -75mm; }
.cover > * { position: relative; z-index: 1; }
.cover .kicker { color: #C59A3D; letter-spacing: .18em; font-size: 9pt; font-weight: 700; margin-bottom: 16mm; }
.cover h1 { max-width: 165mm; font-size: 30pt; line-height: 1.25; margin: 0; color: #17365D; }
.cover .subtitle { color: #2E74B5; font-size: 15pt; margin-top: 8mm; }
.cover .rule { width: 55mm; border-top: 1.2mm solid #C59A3D; margin: 12mm 0 9mm; }
.cover .meta { color: #66737F; font-size: 10pt; line-height: 1.9; }
.cover .conclusion { max-width: 158mm; padding: 5mm 7mm; margin-top: 15mm; background: #F4F7FA; border-left: 1.5mm solid #2E74B5; color: #1F4D78; font-weight: 700; text-align: left; border-radius: 1.5mm; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
h1.part { page-break-before: always; color: #17365D; font-size: 21pt; border-bottom: 1.2mm solid #C59A3D; padding: 0 0 4mm; margin: 0 0 7mm; }
h1.part + h2.talk-page { page-break-before: avoid; }
h2 { color: #2E74B5; font-size: 16pt; line-height: 1.3; margin: 8mm 0 4mm; }
h2.talk-page { page-break-before: always; color: #17365D; font-size: 20pt; padding: 0 0 4mm 5mm; border-left: 2mm solid #2E74B5; border-bottom: .3mm solid #D8E1EA; margin-top: 0; }
h2.usage { color: #17365D; border-bottom: .5mm solid #C59A3D; padding-bottom: 2mm; }
h3 { color: #1F4D78; font-size: 12pt; margin: 5mm 0 2mm; }
h3.screen-label, h3.speaker-label, h3.transition-label { display: inline-block; color: white; padding: 1.2mm 3mm; border-radius: 1.5mm; font-size: 9.5pt; letter-spacing: .04em; }
h3.screen-label { background: #2E74B5; }
h3.speaker-label { background: #C59A3D; }
h3.transition-label { background: #66737F; }
p { margin: 0 0 3mm; text-align: justify; orphans: 3; widows: 3; }
ul, ol { margin: 1mm 0 4mm 7mm; padding-left: 5mm; }
li { margin: 0 0 1.2mm; }
blockquote { margin: 4mm 0; padding: 3.5mm 5mm; background: #EEF4FA; border-left: 1.4mm solid #2E74B5; color: #17365D; break-inside: avoid; page-break-inside: avoid; border-radius: 0 1.5mm 1.5mm 0; }
blockquote p { margin: 0; font-weight: 700; }
pre { background: #17365D; color: #F7FAFC; padding: 4mm 5mm; border-radius: 2mm; font-family: Consolas, monospace; font-size: 9pt; line-height: 1.45; white-space: pre-wrap; break-inside: avoid; page-break-inside: avoid; }
code { font-family: Consolas, monospace; color: #1F4D78; background: #EEF2F5; padding: 0 .8mm; border-radius: .6mm; }
pre code { color: inherit; background: transparent; padding: 0; }
.formula { margin: 4mm auto; padding: 3.5mm; background: #F7F9FB; border: .35mm solid #C9D2DC; border-radius: 1.5mm; text-align: center; color: #17365D; font-family: Cambria, serif; font-size: 12pt; break-inside: avoid; }
.table-wrap { margin: 3mm 0 5mm; break-inside: avoid; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; table-layout: auto; font-size: 8.5pt; line-height: 1.28; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { background: #17365D; color: #fff; font-weight: 700; }
th, td { border: .25mm solid #CBD4DD; padding: 1.7mm 2mm; vertical-align: middle; overflow-wrap: anywhere; }
tbody tr:nth-child(even) { background: #F6F8FA; }
td:not(:first-child), th:not(:first-child) { text-align: center; }
figure { margin: 5mm auto 6mm; text-align: center; break-inside: avoid; page-break-inside: avoid; }
figure img { display: block; max-width: 100%; max-height: 145mm; object-fit: contain; margin: 0 auto; border: .25mm solid #D8E1EA; border-radius: 1.5mm; }
figcaption { margin-top: 2mm; color: #66737F; font-size: 8.5pt; }
figure + p em { color: #66737F; font-size: 8.7pt; }
hr { border: 0; border-top: .35mm solid #D8E1EA; margin: 7mm 0; }
strong { color: #17365D; }
a { color: #2E74B5; text-decoration: none; }
.footer { position: fixed; bottom: -10mm; left: 0; right: 0; color: #8A949C; font-size: 7.5pt; border-top: .2mm solid #D8E1EA; padding-top: 1.5mm; }
@media print {
  .cover { height: 264mm; }
  a { color: inherit; }
}
</style>
</head>
<body>
<section class="cover">
  <div class="kicker">PAPER REPRODUCTION · GROUP MEETING</div>
  <h1>MS-CADM 原论文复现</h1>
  <div class="subtitle">从“代码能运行”到“结论能重现”</div>
  <div class="rule"></div>
  <div class="meta">GEFCom2014 Wind · 10 Zones · Conditional Diffusion<br>建议汇报时长：15–20 分钟<br>整理日期：2026-08-12</div>
  <div class="conclusion">核心结论：方法链路已完整重建，部分定性结论得到支持；核心数值与领先排名没有复现，重实现存在明显欠离散。</div>
</section>
<div class="footer">MS-CADM 原论文复现 · 组会讲稿</div>
${body}
</body>
</html>`;

fs.writeFileSync(htmlOutput, html, "utf8");

const linuxArgs = [
  "--headless=new",
  "--no-sandbox",
  "--disable-gpu",
  "--disable-dev-shm-usage",
  "--allow-file-access-from-files",
  "--no-pdf-header-footer",
  `--print-to-pdf=${pdfOutput}`,
  pathToFileURL(htmlOutput).href,
];
let result = fs.existsSync(linuxChrome)
  ? spawnSync(linuxChrome, linuxArgs, { encoding: "utf8", timeout: 120000 })
  : { status: 1, stderr: "Linux Chromium not found" };

const windowsTempDir = process.env.WINDOWS_TEMP_DIR;
const mountedTempDir = process.env.WSL_WINDOWS_TEMP_DIR;
if (result.status !== 0 && fs.existsSync(windowsChrome) && windowsTempDir && mountedTempDir) {
  const windowsTemp = path.win32.join(windowsTempDir, "MSCADM_GROUP_MEETING_GUIDE.pdf");
  const mountedTemp = path.join(mountedTempDir, "MSCADM_GROUP_MEETING_GUIDE.pdf");
  const windowsPreview = path.win32.join(windowsTempDir, "MSCADM_GROUP_MEETING_GUIDE_cover.png");
  const mountedPreview = path.join(mountedTempDir, "MSCADM_GROUP_MEETING_GUIDE_cover.png");
  const distro = process.env.WSL_DISTRO_NAME || "Ubuntu";
  const windowsHtmlUrl = `file://wsl.localhost/${distro}${htmlOutput.split(path.sep).join("/")}`;
  const windowsArgs = [
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    `--print-to-pdf=${windowsTemp}`,
    windowsHtmlUrl,
  ];
  result = spawnSync(windowsChrome, windowsArgs, { encoding: "utf8", timeout: 120000 });
  if (result.status === 0 && fs.existsSync(mountedTemp)) {
    fs.copyFileSync(mountedTemp, pdfOutput);
    const previewResult = spawnSync(windowsChrome, [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--window-size=1240,1754",
      `--screenshot=${windowsPreview}`,
      windowsHtmlUrl,
    ], { encoding: "utf8", timeout: 120000 });
    if (previewResult.status === 0 && fs.existsSync(mountedPreview)) {
      fs.mkdirSync(path.dirname(previewOutput), { recursive: true });
      fs.copyFileSync(mountedPreview, previewOutput);
    }
    for (const pageNumber of [10, 11]) {
      const name = `MSCADM_GROUP_MEETING_GUIDE_page${pageNumber}.png`;
      const windowsPagePreview = path.win32.join(windowsTempDir, name);
      const mountedPagePreview = path.join(mountedTempDir, name);
      const pageResult = spawnSync(windowsChrome, [
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1240,1754",
        `--screenshot=${windowsPagePreview}`,
        `${windowsHtmlUrl}#page-${pageNumber}`,
      ], { encoding: "utf8", timeout: 120000 });
      if (pageResult.status === 0 && fs.existsSync(mountedPagePreview)) {
        fs.copyFileSync(mountedPagePreview, path.join(here, "qa", name));
      }
    }
  }
}

if (result.status !== 0) {
  process.stderr.write(result.stderr || result.stdout || "Chromium PDF rendering failed\n");
  process.exit(result.status ?? 1);
}
if (!fs.existsSync(pdfOutput) || fs.statSync(pdfOutput).size < 100000) {
  throw new Error(`PDF was not created correctly: ${pdfOutput}`);
}
console.log(JSON.stringify({
  html: htmlOutput,
  pdf: pdfOutput,
  preview: fs.existsSync(previewOutput) ? previewOutput : null,
  bytes: fs.statSync(pdfOutput).size,
}, null, 2));
