import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { marked } from "file:///C:/Users/86177/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const source = path.join(here, "MSCADM_PAPER_REPRODUCTION_REPORT.md");
const output = path.join(here, "qa", "mscadm_paper_preview.html");

const raw = fs.readFileSync(source, "utf8");
const [coverRaw, bodyRaw0] = raw.split("[[COVER_END]]");
const coverLines = coverRaw.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
let bodyRaw = bodyRaw0.replace("[[END_REPORT]]", "");
bodyRaw = bodyRaw.replace("\n## 8.7 Table 5：RTS-24 SUC 重构", "\n<div class=\"page-break\"></div>\n\n## 8.7 Table 5：RTS-24 SUC 重构");

bodyRaw = bodyRaw.replace(/\[\[FIGURE:([^|]+)\|([^\]]+)\]\]/g, (_m, rel, cap) => {
  const uri = pathToFileURL(path.join(root, rel)).href;
  return `\n<figure><img src="${uri}" alt="${cap}"><figcaption>${cap}</figcaption></figure>\n`;
});

bodyRaw = bodyRaw.replace(/\[\[CUSTOM_FIGURE:([^|]+)\|([^\]]+)\]\]/g, (_m, name, cap) => {
  const files = {
    architecture: "mscadm_architecture.png",
    diffusion_workflow: "diffusion_workflow.png",
    reproduction_scorecard: "reproduction_scorecard.png",
    training_curve: "training_curve.png",
  };
  if (!(name in files)) throw new Error(`Unknown custom figure: ${name}`);
  const uri = pathToFileURL(path.join(here, "assets", "mscadm_paper", files[name])).href;
  return `\n<figure><img src="${uri}" alt="${cap}"><figcaption>${cap}</figcaption></figure>\n`;
});

marked.setOptions({ gfm: true, breaks: false });
const bodyHtml = marked.parse(bodyRaw);
const title = coverLines[0].replace(/^#\s*/, "");
const subtitle = coverLines[1] ?? "";
const meta = coverLines.slice(2).map((x) => `<div>${x}</div>`).join("");

const css = `
@page { size: Letter portrait; margin: 0.82in 0.86in 0.78in 0.86in; }
* { box-sizing: border-box; }
body { margin: 0; color: #263238; font-family: Calibri, "Microsoft YaHei", sans-serif; font-size: 10.6pt; line-height: 1.42; }
.cover { height: 9.35in; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; page-break-after: always; }
.kicker { color: #C59A3D; font-size: 10pt; font-weight: 700; letter-spacing: 0.12em; margin-bottom: 0.28in; }
.cover h1 { page-break-before: auto; color: #17365D; font-size: 29pt; line-height: 1.18; margin: 0 0 0.18in; }
.cover .subtitle { color: #2E74B5; font-size: 15pt; margin-bottom: 0.30in; }
.rule { width: 68%; border-top: 2px solid #C59A3D; margin: 0.05in 0 0.28in; }
.meta { color: #66737F; font-size: 10pt; line-height: 1.8; }
.badge { margin-top: 0.32in; padding: 0.10in 0.18in; background: #EAF2F8; color: #1F4D78; font-size: 9.5pt; font-weight: 700; border-radius: 6px; }
.page-break { break-after: page; page-break-after: always; }
h1 { color: #2E74B5; font-size: 16pt; line-height: 1.22; margin: 0; padding-top: 0.06in; page-break-before: always; page-break-after: avoid; }
h2 { color: #2E74B5; font-size: 13pt; line-height: 1.25; margin: 0.16in 0 0.08in; page-break-after: avoid; }
h3 { color: #1F4D78; font-size: 12pt; line-height: 1.25; margin: 0.12in 0 0.06in; page-break-after: avoid; }
p { margin: 0 0 0.10in; text-align: justify; orphans: 3; widows: 3; break-inside: avoid; }
ul, ol { margin: 0.04in 0 0.12in 0.32in; padding-left: 0.14in; }
li { margin-bottom: 0.045in; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 0.10in 0 0.16in; font-size: 8.1pt; line-height: 1.22; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #17365D; color: white; font-weight: 700; text-align: center; }
th, td { border: 0.6px solid #C9D2DC; padding: 5px 6px; vertical-align: middle; overflow-wrap: anywhere; }
tbody tr:nth-child(even) { background: #F7F9FB; }
td:not(:first-child) { text-align: center; }
figure { margin: 0.14in 0 0.18in; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 5.7in; object-fit: contain; }
figcaption { margin-top: 0.05in; color: #66737F; font-size: 8.8pt; font-style: italic; line-height: 1.28; }
code { font-family: Consolas, monospace; color: #1F4D78; background: #EEF2F5; padding: 0 2px; }
`;

const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>${title}</title><style>${css}</style></head><body>
<section class="cover">
  <div class="kicker">PAPER METHOD · ARCHITECTURE · REPRODUCTION AUDIT · 2026</div>
  <h1>${title}</h1>
  <div class="subtitle">${subtitle}</div>
  <div class="rule"></div>
  <div class="meta">${meta}</div>
  <div class="badge">论文原理 · 网络张量流 · 训练与采样 · 指标复现 · SUC · 漏洞审计</div>
</section>
${bodyHtml}
</body></html>`;

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, html, "utf8");
console.log(output);
