import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const input = path.join(here, "MSCADM_GROUP_MEETING_GUIDE.html");
const output = path.join(here, "MSCADM_GROUP_MEETING_GUIDE_STANDALONE.html");

if (!fs.existsSync(input)) {
  throw new Error(`Missing generated HTML source: ${input}`);
}

let html = fs.readFileSync(input, "utf8");

// Embed every local image so this one HTML file can be copied anywhere.
html = html.replace(/<img src="([^"]+)"/g, (match, src) => {
  if (/^(data:|https?:)/.test(src)) return match;
  const absolute = path.resolve(here, src);
  if (!fs.existsSync(absolute)) {
    throw new Error(`Missing image: ${absolute}`);
  }
  const extension = path.extname(absolute).toLowerCase();
  const mime = extension === ".jpg" || extension === ".jpeg" ? "image/jpeg" : "image/png";
  const encoded = fs.readFileSync(absolute).toString("base64");
  return `<img src="data:${mime};base64,${encoded}"`;
});

const diffusionMath = `<div class="formula math-card" aria-label="x t equals square root alpha bar t times x zero plus square root one minus alpha bar t times epsilon">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <msub><mi>x</mi><mi>t</mi></msub><mo>=</mo>
    <msqrt><msub><mover><mi>α</mi><mo>¯</mo></mover><mi>t</mi></msub></msqrt>
    <msub><mi>x</mi><mn>0</mn></msub><mo>+</mo>
    <msqrt><mrow><mn>1</mn><mo>−</mo><msub><mover><mi>α</mi><mo>¯</mo></mover><mi>t</mi></msub></mrow></msqrt>
    <mi>ε</mi>
  </mrow>
</math>
</div>`;

const adalnMath = `<div class="formula math-card" aria-label="h is updated by h plus g t times F of one plus s t times layer normalization h plus b t">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>h</mi><mo>←</mo><mi>h</mi><mo>+</mo><msub><mi>g</mi><mi>t</mi></msub><mo>·</mo>
    <mi>F</mi><mo>(</mo>
    <mo>(</mo><mn>1</mn><mo>+</mo><msub><mi>s</mi><mi>t</mi></msub><mo>)</mo>
    <mi mathvariant="normal">LN</mi><mo>(</mo><mi>h</mi><mo>)</mo><mo>+</mo><msub><mi>b</mi><mi>t</mi></msub>
    <mo>)</mo>
  </mrow>
</math>
</div>`;

html = html.replace(
  /<div class="formula">x_t=.*?<\/div>/s,
  diffusionMath,
);
html = html.replace(
  /<div class="formula">h \\leftarrow.*?<\/div>/s,
  adalnMath,
);

const navigation = `<nav class="screen-nav" aria-label="组会讲稿导航">
  <a href="#top" class="brand">MS-CADM 复现</a>
  <div class="nav-pages">
    ${Array.from({ length: 14 }, (_, index) => `<a href="#page-${index + 1}">${index + 1}</a>`).join("")}
  </div>
  <button id="theme-toggle" type="button" aria-label="切换深色模式">深色</button>
</nav>
<div class="browser-tip">单文件浏览器版 · 图片已内嵌 · 公式使用原生 MathML · 按 <kbd>Ctrl</kbd> + <kbd>P</kbd> 可打印</div>`;

html = html.replace("<body>", `<body id="top">${navigation}`);
html = html.replace("</body>", `<script>
const button = document.getElementById("theme-toggle");
button.addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
  button.textContent = document.documentElement.classList.contains("dark") ? "浅色" : "深色";
});
</script></body>`);

const screenCss = `
<style>
.screen-nav, .browser-tip { display: none; }
.math-card math { font-family: "Cambria Math", "STIX Two Math", serif; font-size: 1.42em; }
@media screen {
  html { scroll-behavior: smooth; background: #E9EEF3; }
  body { max-width: 1120px; margin: 0 auto; padding: 58px 46px 80px; background: #FFFFFF; box-shadow: 0 0 32px rgba(28, 50, 73, .12); }
  .screen-nav { display: flex; position: fixed; z-index: 100; top: 0; left: 0; right: 0; height: 46px; align-items: center; gap: 18px; padding: 0 24px; background: rgba(23,54,93,.96); color: white; box-shadow: 0 2px 8px rgba(0,0,0,.18); }
  .screen-nav a { color: white; }
  .screen-nav .brand { font-weight: 700; white-space: nowrap; }
  .nav-pages { display: flex; gap: 5px; flex: 1; justify-content: center; }
  .nav-pages a { width: 25px; height: 25px; display: grid; place-items: center; border-radius: 50%; font-size: 8.5pt; background: rgba(255,255,255,.10); }
  .nav-pages a:hover { background: #C59A3D; }
  #theme-toggle { border: 1px solid rgba(255,255,255,.55); color: white; background: transparent; border-radius: 14px; padding: 3px 11px; cursor: pointer; font-family: inherit; }
  .browser-tip { display: block; margin: 0 0 18px; padding: 10px 14px; color: #1F4D78; background: #EAF2F8; border-radius: 6px; font-size: 9pt; text-align: center; }
  kbd { padding: 1px 5px; border: 1px solid #B8C3CE; border-bottom-width: 2px; border-radius: 4px; background: white; }
  .cover { min-height: calc(100vh - 104px); height: auto; max-height: 920px; margin-bottom: 42px; border-radius: 10px; background: white; }
  h1.part { page-break-before: auto; margin-top: 55px; scroll-margin-top: 62px; }
  h2.talk-page { page-break-before: auto; margin-top: 54px; padding-top: 12px; scroll-margin-top: 58px; }
  h2.talk-page::before { content: "讲述单元"; display: block; color: #C59A3D; font-size: 8pt; letter-spacing: .15em; margin-bottom: 5px; }
  figure { margin: 28px auto 34px; }
  figure img { max-height: 680px; box-shadow: 0 7px 24px rgba(31,77,120,.10); }
  .math-card { font-size: 14pt; padding: 16px 20px; overflow-x: auto; }
  .footer { display: none; }
}
@media screen and (max-width: 760px) {
  body { padding: 54px 18px 60px; }
  .nav-pages { display: none; }
  .screen-nav { padding: 0 14px; }
  .screen-nav .brand { flex: 1; }
  .cover h1 { font-size: 25pt; }
  table { font-size: 8pt; }
  .table-wrap { overflow-x: auto; }
}
html.dark { background: #111820; }
html.dark body { background: #18222D; color: #E7EDF3; }
html.dark .cover { background: #18222D; }
html.dark h1, html.dark h2, html.dark h3, html.dark strong { color: #8EC5F4; }
html.dark p, html.dark li, html.dark td { color: #E7EDF3; }
html.dark blockquote, html.dark .formula, html.dark pre, html.dark .browser-tip { background: #223244; color: #D9EBFA; }
html.dark tbody tr:nth-child(even) { background: #22303D; }
html.dark code { background: #263847; color: #A8D4F8; }
html.dark kbd { background: #273541; color: white; }
@media print {
  .screen-nav, .browser-tip { display: none !important; }
}
</style>`;

html = html.replace("</head>", `${screenCss}</head>`);
fs.writeFileSync(output, html, "utf8");

console.log(JSON.stringify({
  output,
  bytes: fs.statSync(output).size,
  embeddedImages: (html.match(/src="data:image\//g) || []).length,
  mathmlBlocks: (html.match(/<math display="block"/g) || []).length,
}, null, 2));
