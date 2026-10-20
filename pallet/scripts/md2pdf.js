// Reproduce the Markdown PDF (yzane) output for our docs: markdown-it + container + highlight, Letter, user margins, page-number
// footer, no default styles (as in the user's settings.json), printed by Chrome through puppeteer-core.
const path = require('path'), fs = require('fs');
const EXT = process.env.HOME + '/.vscode/extensions/yzane.markdown-pdf-1.5.0/node_modules/';
const hljs = require(EXT + 'highlight.js');
const md = require(EXT + 'markdown-it')({ html: true, breaks: false, linkify: true,
  highlight: (str, lang) => { if (lang && hljs.getLanguage(lang)) { try { return '<pre class="hljs"><code>' + hljs.highlight(lang, str, true).value + '</code></pre>'; } catch (e) {} }
    return '<pre class="hljs"><code>' + md.utils.escapeHtml(str) + '</code></pre>'; } });
md.use(require(EXT + 'markdown-it-container'), '', {   // same as the extension: any ::: name becomes <div class="name">
  validate: () => true,
  render: (tokens, idx) => tokens[idx].nesting === 1 ? '<div class="' + tokens[idx].info.trim() + '">\n' : '</div>\n' });
const [,, srcArg, outArg] = process.argv; const src = path.resolve(srcArg), out = path.resolve(outArg);
const body = md.render(fs.readFileSync(src, 'utf8'));
const hl = fs.readFileSync(EXT + 'highlight.js/styles/atom-one-light.css', 'utf8');
const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${path.basename(src)}</title><style>${hl}</style></head><body>${body}</body></html>`;
const tmp = out + '.html'; fs.writeFileSync(tmp, html);
(async () => {
  const puppeteer = require(EXT + 'puppeteer-core');
  const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', args: ['--no-sandbox', '--disable-gpu'] });
  const page = await browser.newPage();
  await page.goto('file://' + tmp, { waitUntil: 'networkidle0' });
  await page.pdf({ path: out, format: 'Letter', printBackground: true, displayHeaderFooter: true,
    headerTemplate: "<div style='display: none;'></div>",
    footerTemplate: "<div style='font-size: 10pt; font-family: \"Times New Roman\", serif; width: 100%; text-align: center; padding-top: 20px;'><span class='pageNumber'></span></div>",
    margin: { top: '0.75in', bottom: '0.75in', left: '0.6in', right: '0.6in' } });
  await browser.close(); fs.unlinkSync(tmp); console.log('wrote', out);
})().catch(e => { console.error(e); process.exit(1); });
