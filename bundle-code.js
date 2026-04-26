#!/usr/bin/env node

(async function main() {
  const fs = await import("node:fs");
  const fsp = await import("node:fs/promises");
  const path = await import("node:path");
  const crypto = await import("node:crypto");

  const PROJECT_ROOT = process.cwd();
  const IGNORE_FILE_NAME = ".codebundleignore";
  const OUTPUT_FILE_NAME = "codebundle.txt";
  const OUTPUT_FILE = path.join(PROJECT_ROOT, OUTPUT_FILE_NAME);

  const COMPACT_MODE = true;
  const MAX_FILE_SIZE_BYTES = 512 * 1024;
  const MAX_TOTAL_BYTES = 10 * 1024 * 1024;
  const MAX_TREE_DEPTH = 8;
  const MAX_TREE_ENTRIES = 2000;

  const ONLY_EXTENSIONS = new Set([
    // Пусто = любые текстовые файлы.
  ]);

  const IMPORTANT_FILES = [
    "README.md",
    "readme.md",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "vite.config.js",
    "vite.config.ts",
    "main.py",
    "app.py",
    "server.js",
    "index.js",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "neat_config.ini",
  ];

  const DEFAULT_IGNORE = [
    ".git/",
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".nuxt/",
    ".svelte-kit/",
    ".vite/",
    "coverage/",
    "htmlcov/",
    "venv/",
    ".venv/",
    "env/",
    "ENV/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".idea/",
    ".vscode/",
    ".DS_Store",
    "Thumbs.db",
    OUTPUT_FILE_NAME,
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.rar",
    "*.7z",
    "*.exe",
    "*.dll",
    "*.dylib",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp4",
    "*.mp3",
    "*.wav",
    "*.pkl",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.ckpt",
  ];

  const LANG_BY_EXT = {
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".py": "python",
    ".json": "json",
    ".md": "markdown",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".sh": "bash",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
  };

  function normalizeToPosix(p) {
    return p.split(path.sep).join("/");
  }

  function stripBom(s) {
    return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s;
  }

  function escapeRegex(s) {
    return s.replace(/[|\\{}()[\]^$+?.]/g, "\\$&");
  }

  function globToRegex(pattern) {
    let p = normalizeToPosix(pattern.trim());
    if (p.startsWith("./")) p = p.slice(2);

    const directoryOnly = p.endsWith("/");
    if (directoryOnly) p = p.slice(0, -1);

    let out = "";

    for (let i = 0; i < p.length; i++) {
      const ch = p[i];

      if (ch === "*") {
        if (p[i + 1] === "*") {
          out += ".*";
          i++;
        } else {
          out += "[^/]*";
        }
      } else if (ch === "?") {
        out += "[^/]";
      } else {
        out += escapeRegex(ch);
      }
    }

    if (!p.includes("/")) out = `(^|.*/)${out}`;
    else out = `^${out}`;

    out += directoryOnly ? `(/.*)?$` : `$`;
    return new RegExp(out);
  }

  async function readIgnoreFile(ignorePath) {
    try {
      const raw = await fsp.readFile(ignorePath, "utf8");
      return stripBom(raw)
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(line => line && !line.startsWith("#"));
    } catch {
      return [];
    }
  }

  function buildIgnoreMatcher(patterns) {
    const rules = [...DEFAULT_IGNORE, ...patterns]
      .map(raw => {
        const neg = raw.startsWith("!");
        const pat = neg ? raw.slice(1).trim() : raw.trim();
        return pat ? { neg, re: globToRegex(pat) } : null;
      })
      .filter(Boolean);

    return relPosix => {
      let ignored = false;

      for (const rule of rules) {
        if (rule.re.test(relPosix)) ignored = !rule.neg;
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
          b === 9 ||
          b === 10 ||
          b === 13 ||
          (b >= 32 && b <= 126) ||
          b >= 128;

        if (!ok) weirdCount++;
      }

      return nulCount > 0 || weirdCount / bytesRead > 0.25;
    } finally {
      await fd.close();
    }
  }

  function guessLanguage(relPosix) {
    const base = path.basename(relPosix);
    const ext = path.extname(relPosix).toLowerCase();

    if (base === "Dockerfile") return "dockerfile";
    if (base === ".gitignore" || base === ".codebundleignore") return "gitignore";

    return LANG_BY_EXT[ext] || "text";
  }

  function countLines(content) {
    return content ? content.split(/\r\n|\r|\n/).length : 0;
  }

  function sha1Short(content) {
    return crypto.createHash("sha1").update(content).digest("hex").slice(0, 12);
  }

  function looksSensitive(relPosix, content) {
    const name = path.basename(relPosix).toLowerCase();

    if (name === ".env" || name.startsWith(".env.")) return true;

    return [
      /api[_-]?key\s*=\s*["']?[A-Za-z0-9_\-]{20,}/i,
      /secret\s*=\s*["']?[A-Za-z0-9_\-]{20,}/i,
      /token\s*=\s*["']?[A-Za-z0-9_\-]{20,}/i,
      /password\s*=\s*["']?.{8,}/i,
      /-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----/,
    ].some(re => re.test(content));
  }

  function compactContent(content) {
    let s = stripBom(content);

    s = s.replace(/[ \t]+$/gm, "");
    s = s.replace(/^\s*#\s*[=\-*#_]{5,}\s*$/gm, "");
    s = s.replace(/^\s*\/\/\s*[=\-*#_]{5,}\s*$/gm, "");
    s = s.replace(/^\s*\/\*\s*[=\-*#_]{5,}\s*\*\/\s*$/gm, "");
    s = s.replace(/\n{4,}/g, "\n\n\n");

    return s;
  }

  async function* walk(dirAbs, isIgnored) {
    let entries;

    try {
      entries = await fsp.readdir(dirAbs, { withFileTypes: true });
    } catch {
      return;
    }

    entries.sort((a, b) => {
      if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    for (const ent of entries) {
      const abs = path.join(dirAbs, ent.name);
      const rel = path.relative(PROJECT_ROOT, abs);
      const relPosix = normalizeToPosix(rel);

      if (!relPosix) continue;
      if (path.resolve(abs) === path.resolve(OUTPUT_FILE)) continue;
      if (isIgnored(relPosix)) continue;

      if (ent.isDirectory()) {
        yield* walk(abs, isIgnored);
      } else if (ent.isFile()) {
        yield { abs, relPosix };
      }
    }
  }

  function filePriority(relPosix) {
    const base = path.basename(relPosix);

    if (IMPORTANT_FILES.includes(base) || IMPORTANT_FILES.includes(relPosix)) return 0;
    if (relPosix.includes("/src/")) return 1;
    if (relPosix.includes("/server/")) return 1;
    if (relPosix.includes("/core/")) return 1;
    if (relPosix.includes("/config/")) return 1;
    if (relPosix.endsWith(".py")) return 2;
    if (/\.(js|jsx|ts|tsx)$/.test(relPosix)) return 2;
    if (/\.(json|ini|yml|yaml|toml)$/.test(relPosix)) return 3;

    return 5;
  }

  async function collectFiles(isIgnored) {
    const files = [];

    const skipped = {
      ext: 0,
      size: 0,
      binary: 0,
      unreadable: 0,
      sensitive: 0,
    };

    for await (const file of walk(PROJECT_ROOT, isIgnored)) {
      const ext = path.extname(file.relPosix).toLowerCase();

      if (ONLY_EXTENSIONS.size && !ONLY_EXTENSIONS.has(ext)) {
        skipped.ext++;
        continue;
      }

      let stat;

      try {
        stat = await fsp.stat(file.abs);
      } catch {
        skipped.unreadable++;
        continue;
      }

      if (stat.size > MAX_FILE_SIZE_BYTES) {
        skipped.size++;
        continue;
      }

      let binary;

      try {
        binary = await isProbablyBinary(file.abs);
      } catch {
        skipped.unreadable++;
        continue;
      }

      if (binary) {
        skipped.binary++;
        continue;
      }

      let content;

      try {
        content = await fsp.readFile(file.abs, "utf8");
        content = COMPACT_MODE ? compactContent(content) : stripBom(content);
      } catch {
        skipped.unreadable++;
        continue;
      }

      if (looksSensitive(file.relPosix, content)) {
        skipped.sensitive++;
        continue;
      }

      files.push({
        ...file,
        content,
        bytes: Buffer.byteLength(content, "utf8"),
        lines: countLines(content),
        lang: guessLanguage(file.relPosix),
        sha1: sha1Short(content),
        priority: filePriority(file.relPosix),
      });
    }

    files.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return a.relPosix.localeCompare(b.relPosix);
    });

    return { files, skipped };
  }

  function buildTree(files) {
    const root = {};

    for (const file of files) {
      const parts = file.relPosix.split("/");
      let node = root;

      for (const part of parts) {
        node[part] ||= {};
        node = node[part];
      }
    }

    const lines = [];
    let entries = 0;

    function render(node, prefix = "", depth = 0) {
      if (depth > MAX_TREE_DEPTH || entries > MAX_TREE_ENTRIES) return;

      const names = Object.keys(node).sort((a, b) => {
        const ad = Object.keys(node[a]).length > 0;
        const bd = Object.keys(node[b]).length > 0;

        if (ad !== bd) return ad ? -1 : 1;
        return a.localeCompare(b);
      });

      names.forEach((name, index) => {
        if (entries > MAX_TREE_ENTRIES) return;

        const isLast = index === names.length - 1;
        const child = node[name];
        const isDir = Object.keys(child).length > 0;

        lines.push(`${prefix}${isLast ? "└" : "├"} ${name}${isDir ? "/" : ""}`);
        entries++;

        if (isDir) render(child, `${prefix}${isLast ? "  " : "│ "}`, depth + 1);
      });
    }

    render(root);

    if (entries > MAX_TREE_ENTRIES) {
      lines.push(`...tree truncated after ${MAX_TREE_ENTRIES} entries`);
    }

    return lines.join("\n");
  }

  function writeFileSection(out, file, index) {
    out.write(`\n<<<FILE ${index + 1} /${file.relPosix} ${file.lang} ${file.lines}l ${file.bytes}b ${file.sha1}>>>\n`);
    out.write(file.content);
    if (!file.content.endsWith("\n")) out.write("\n");
    out.write(`<<<END FILE>>>\n`);
  }

  async function bundle() {
    const ignorePath = path.join(PROJECT_ROOT, IGNORE_FILE_NAME);
    const patterns = await readIgnoreFile(ignorePath);
    const isIgnored = buildIgnoreMatcher(patterns);

    const { files, skipped } = await collectFiles(isIgnored);
    const out = fs.createWriteStream(OUTPUT_FILE, { encoding: "utf8" });

    let totalBytes = 0;
    let included = 0;
    let stoppedByLimit = false;

    out.write(`GPT_CODE_BUNDLE\n`);
    out.write(`root=${normalizeToPosix(PROJECT_ROOT)}\n`);
    out.write(`generated=${new Date().toISOString()}\n`);
    out.write(`ignore=${IGNORE_FILE_NAME}\n`);
    out.write(`compact=${COMPACT_MODE}\n`);
    out.write(`max_file_bytes=${MAX_FILE_SIZE_BYTES}\n`);
    out.write(`max_total_bytes=${MAX_TOTAL_BYTES}\n`);

    out.write(`\nTREE\n`);
    out.write(buildTree(files));
    out.write(`\n`);

    out.write(`\nMANIFEST\n`);
    files.forEach((file, i) => {
      out.write(
        `${String(i + 1).padStart(3, "0")} /${file.relPosix} ${file.lang} ${file.lines}l ${file.bytes}b ${file.sha1}\n`
      );
    });

    out.write(`\nFILES\n`);

    for (const file of files) {
      if (totalBytes + file.bytes > MAX_TOTAL_BYTES) {
        stoppedByLimit = true;
        out.write(`\nSTOP total_size_limit=${MAX_TOTAL_BYTES}\n`);
        break;
      }

      writeFileSection(out, file, included);
      totalBytes += file.bytes;
      included++;
    }

    out.write(`\nSUMMARY\n`);
    out.write(`included_files=${included}\n`);
    out.write(`candidate_files=${files.length}\n`);
    out.write(`total_bytes=${totalBytes}\n`);
    out.write(`stopped_by_limit=${stoppedByLimit}\n`);
    out.write(`skipped_ext=${skipped.ext}\n`);
    out.write(`skipped_size=${skipped.size}\n`);
    out.write(`skipped_binary=${skipped.binary}\n`);
    out.write(`skipped_sensitive=${skipped.sensitive}\n`);
    out.write(`skipped_unreadable=${skipped.unreadable}\n`);

    await new Promise((resolve, reject) => {
      out.end(resolve);
      out.on("error", reject);
    });

    console.log(`Done: ${OUTPUT_FILE}`);
    console.log(`Included: ${included}/${files.length}, bytes: ${totalBytes}`);
  }

  await bundle();
})().catch(err => {
  console.error("Error:", err);
  process.exitCode = 1;
});