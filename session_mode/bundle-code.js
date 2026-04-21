#!/usr/bin/env node
/**
 * bundle-code.js
 * Собирает весь текстовый код проекта в один файл, помечая секции путями.
 * - Рекурсивный обход
 * - Пропуск бинарных файлов (по содержимому)
 * - Игнор по .codebundleignore (gitignore-like)
 */







const  fs = require("fs");
const  fsp = require("fs/promises");
const  path = require("path")

const PROJECT_ROOT = process.cwd();
const IGNORE_FILE_NAME = ".codebundleignore";
const OUTPUT_FILE = path.join(PROJECT_ROOT, "codebundle.txt");

const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024; // 5MB
const MAX_TOTAL_BYTES = 200 * 1024 * 1024;   // 200MB

// Если заполнить — соберёт только эти расширения. Если пусто — любые текстовые.
const ONLY_EXTENSIONS = new Set([
  // ".js", ".jsx", ".ts", ".tsx", ".json", ".css", ".scss", ".md", ".html", ".yml", ".yaml"
]);

function normalizeToPosix(p) {
  return p.split(path.sep).join("/");
}

function stripBom(s) {
  return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s;
}

async function readIgnoreFile(ignorePath) {
  try {
    const raw = await fsp.readFile(ignorePath, "utf8");
    return stripBom(raw)
      .split(/\r?\n/)
      .map(l => l.trim())
      .filter(l => l && !l.startsWith("#"));
  } catch {
    return [];
  }
}

/**
 * Компилируем игнор-правила.
 * Поддержка:
 * - "node_modules/" (игнор любой сегмент /node_modules/ в любом месте)
 * - "dist/" и т.п.
 * - "*.ext"
 * - "path/to/file" (точный относительный путь)
 * - "!pattern" (снять игнор)
 */
function buildIgnoreMatcher(patterns) {
  // дефолтные, чтобы не протекало даже если забыли
  const defaultPatterns = [
    "node_modules/",
    ".git/",
    "dist/",
    "build/",
    ".next/",
    "coverage/",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
  ];

  const all = [...defaultPatterns, ...patterns];

  const rules = all.map(p => ({
    neg: p.startsWith("!"),
    pat: (p.startsWith("!") ? p.slice(1) : p).trim(),
  })).filter(r => r.pat);

  function matchRule(relPosix, pat) {
    // relPosix: "admin/node_modules/express/index.js"

    // directory pattern: "node_modules/"
    if (pat.endsWith("/")) {
      const seg = pat.slice(0, -1);
      if (!seg) return false;

      // матч:
      // - если путь начинается с seg/
      // - или содержит /seg/ как сегмент
      return (
        relPosix === seg ||
        relPosix.startsWith(seg + "/") ||
        relPosix.includes("/" + seg + "/") ||
        relPosix.endsWith("/" + seg)
      );
    }

    // glob extension: "*.png"
    if (pat.startsWith("*.") && !pat.includes("/")) {
      const ext = pat.slice(1); // ".png"
      return relPosix.toLowerCase().endsWith(ext.toLowerCase());
    }

    // exact file path
    return relPosix === pat;
  }

  return function isIgnored(relPosix) {
    let ignored = false;
    for (const r of rules) {
      if (matchRule(relPosix, r.pat)) {
        ignored = !r.neg; // если negation — снимаем игнор
      }
    }
    return ignored;
  };
}

async function isProbablyBinary(filePath) {
  const fd = await fsp.open(filePath, "r");
  try {
    const { size } = await fd.stat();
    const toRead = Math.min(size, 8192);
    const buf = Buffer.allocUnsafe(toRead);
    const { bytesRead } = await fd.read(buf, 0, toRead, 0);
    if (bytesRead === 0) return false;

    let nulCount = 0;
    let weirdCount = 0;
    for (let i = 0; i < bytesRead; i++) {
      const b = buf[i];
      if (b === 0) nulCount++;
      const ok =
        b === 9 || b === 10 || b === 13 || (b >= 32 && b <= 126) || b >= 128;
      if (!ok) weirdCount++;
    }
    if (nulCount > 0) return true;
    return (weirdCount / bytesRead) > 0.25;
  } finally {
    await fd.close();
  }
}

async function* walk(dirAbs, isIgnored) {
  let entries;
  try {
    entries = await fsp.readdir(dirAbs, { withFileTypes: true });
  } catch {
    return;
  }

  entries.sort((a, b) => a.name.localeCompare(b.name));

  for (const ent of entries) {
    const abs = path.join(dirAbs, ent.name);
    const rel = path.relative(PROJECT_ROOT, abs);
    const relPosix = normalizeToPosix(rel);

    // не включаем output и ignore файл
    if (path.resolve(abs) === path.resolve(OUTPUT_FILE)) continue;
    if (relPosix === IGNORE_FILE_NAME) continue;

    if (isIgnored(relPosix)) continue;

    if (ent.isDirectory()) {
      yield* walk(abs, isIgnored);
    } else if (ent.isFile()) {
      yield { abs, relPosix };
    }
  }
}

async function bundle() {
  const ignorePath = path.join(PROJECT_ROOT, IGNORE_FILE_NAME);
  const patterns = await readIgnoreFile(ignorePath);
  const isIgnored = buildIgnoreMatcher(patterns);

  const out = fs.createWriteStream(OUTPUT_FILE, { encoding: "utf8" });

  let totalBytes = 0;
  let included = 0;
  let skippedBinary = 0;
  let skippedSize = 0;
  let skippedExt = 0;

  out.write(`=== CODE BUNDLE ===\n`);
  out.write(`root: ${normalizeToPosix(PROJECT_ROOT)}\n`);
  out.write(`ignore: ${IGNORE_FILE_NAME}\n`);
  out.write(`generated: ${new Date().toISOString()}\n\n`);

  for await (const file of walk(PROJECT_ROOT, isIgnored)) {
    const ext = path.extname(file.relPosix).toLowerCase();
    if (ONLY_EXTENSIONS.size && !ONLY_EXTENSIONS.has(ext)) {
      skippedExt++;
      continue;
    }

    let stat;
    try {
      stat = await fsp.stat(file.abs);
    } catch {
      continue;
    }

    if (stat.size > MAX_FILE_SIZE_BYTES) {
      skippedSize++;
      continue;
    }
    if (totalBytes + stat.size > MAX_TOTAL_BYTES) {
      out.write(`\n\n=== STOP: total size limit reached (${MAX_TOTAL_BYTES} bytes) ===\n`);
      break;
    }

    let binary;
    try {
      binary = await isProbablyBinary(file.abs);
    } catch {
      continue;
    }
    if (binary) {
      skippedBinary++;
      continue;
    }

    let content;
    try {
      content = await fsp.readFile(file.abs, "utf8");
    } catch {
      continue;
    }

    out.write(`\n\npath: /${file.relPosix}\n`);
    out.write(`----------------------------------------\n`);
    out.write(content);
    if (!content.endsWith("\n")) out.write("\n");

    totalBytes += stat.size;
    included++;
  }

  out.write(`\n\n=== SUMMARY ===\n`);
  out.write(`included: ${included}\n`);
  out.write(`skipped(binary): ${skippedBinary}\n`);
  out.write(`skipped(size): ${skippedSize}\n`);
  out.write(`skipped(ext): ${skippedExt}\n`);
  out.write(`totalBytes(approx): ${totalBytes}\n`);

  await new Promise((res, rej) => {
    out.end(res);
    out.on("error", rej);
  });

  console.log(`Done. Output: ${OUTPUT_FILE}`);
}

bundle().catch(err => {
  console.error("Error:", err);
  process.exitCode = 1;
});