// Builds the React Event Lab page: node lab/build.cjs
// Output: lab/dist/react-event-lab.html (publish that file as the artifact).
const fs = require("fs");
const path = require("path");
const esbuild = require("esbuild");
const Prism = require("prismjs");
require("prismjs/components/prism-jsx");

const dir = __dirname;

// Newest topic first; the first entry is the default tab.
const exercises = [
  {
    id: "like",
    tab: "useState · Topic 7",
    file: "07-like-button.jsx",
    hint: "Answer the questions in the chat first. Then click the first heart twice, then the second heart. Watch the <b>State</b> panel and which components re-render. The lab put 2 sample items in <code>localStorage</code> for the cart.",
    seedStorage: { cart: ["shoe", "hat"] },
  },
  {
    id: "lazy",
    tab: "Lazy vs eager · Topic 7",
    file: "07-lazy-init.jsx",
    hint: "Click each <b>Re-render</b> button a few times and watch the <b>Console</b>. Both boxes start the same, but only one reads the cart again on every render.",
    seedStorage: { cart: ["shoe", "hat"] },
  },
  {
    id: "shop",
    tab: "Shop · Topic 6",
    file: "06-shop.jsx",
    hint: "Click <b>Add</b> and <b>Remove</b>, click a product name, then type <b>&nbsp;&nbsp;laptop&nbsp;&nbsp;</b> and press Enter.",
  },
  {
    id: "quick",
    tab: "Quick check · Topic 6",
    file: "06-quick-check.jsx",
    hint: "Click <b>Title</b>, <b>Like</b> and <b>Share</b> and compare the bubbling paths.",
  },
];

// Split Prism's HTML into lines, closing and reopening spans across line breaks.
function splitLines(html) {
  const lines = [];
  let cur = "";
  const open = [];
  const re = /<span [^>]*>|<\/span>|\n|[^<\n]+/g;
  let m;
  while ((m = re.exec(html))) {
    const t = m[0];
    if (t === "\n") {
      cur += "</span>".repeat(open.length);
      lines.push(cur);
      cur = open.join("");
    } else if (t.startsWith("<span")) {
      open.push(t);
      cur += t;
    } else if (t === "</span>") {
      open.pop();
      cur += t;
    } else cur += t;
  }
  lines.push(cur);
  return lines;
}

// Find useState names per component, in hook order: { LikeButton: [{ name, setter, line }] }
function stateNames(lines) {
  const out = {};
  let comp = null;
  lines.forEach((l, i) => {
    const f = l.match(/^(?:export\s+)?function\s+([A-Z]\w*)\s*\(/);
    if (f) comp = f[1];
    const s = l.match(/const\s+\[\s*(\w+)\s*(?:,\s*(\w+)\s*)?\]\s*=\s*useState\b/);
    if (s && comp) (out[comp] ||= []).push({ name: s[1], setter: s[2] || null, line: i + 1 });
  });
  return out;
}

// Wrap setter and console.log calls so the page knows which source line ran them.
function markCallSites(lines, names) {
  const setters = Object.values(names).flat().map((s) => s.setter).filter(Boolean);
  const callRe = setters.length ? new RegExp("\\b(" + setters.join("|") + ")\\(", "g") : null;
  return lines
    .map((l, i) => {
      let out = l.replace(/\bconsole\.log\(/g, `__at(${i + 1}, console.log)(`);
      if (callRe) out = out.replace(callRe, `__at(${i + 1}, $1)(`);
      return out;
    })
    .join("\n");
}

const parts = exercises.map((x) => {
  const src = fs.readFileSync(path.join(dir, "exercises", x.file), "utf8").replace(/\n$/, "");
  const lines = src.split("\n");
  const names = stateNames(lines);
  const { code } = esbuild.transformSync(markCallSites(lines, names), {
    loader: "jsx", jsx: "automatic", jsxDev: true, format: "cjs", sourcefile: x.file,
  });
  const html = splitLines(Prism.highlight(src, Prism.languages.jsx, "jsx"));
  const meta = { id: x.id, tab: x.tab, hint: x.hint, seedStorage: x.seedStorage || null, lines, html, stateNames: names };
  return `Object.assign(${JSON.stringify(meta)}, { mount: function (require, __at) {\n${code}\nreturn app;\n} })`;
});

const out = fs.readFileSync(path.join(dir, "template.html"), "utf8")
  .replace("/*__EXERCISES__*/", () => "const EXERCISES = [\n" + parts.join(",\n") + "\n];");
fs.mkdirSync(path.join(dir, "dist"), { recursive: true });
fs.writeFileSync(path.join(dir, "dist", "react-event-lab.html"), out);
console.log("built lab/dist/react-event-lab.html,", out.length, "bytes");
