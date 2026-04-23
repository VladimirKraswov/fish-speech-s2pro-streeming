#!/usr/bin/env node

const http = require("http");
const fs = require("fs");
const path = require("path");
const url = require("url");

const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 4173);
const ROOT = path.resolve(process.argv[2] || ".");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".cjs": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".ico": "image/x-icon",
};

function send(res, status, body, contentType = "text/plain; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function safeResolve(root, pathname) {
  const decoded = decodeURIComponent(pathname);
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  const resolved = path.resolve(root, "." + normalized);
  if (!resolved.startsWith(root)) {
    return null;
  }
  return resolved;
}

function maybeDirectoryIndex(filePath) {
  try {
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      return path.join(filePath, "index.html");
    }
    return filePath;
  } catch {
    return filePath;
  }
}

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url);
  const pathname = parsed.pathname || "/";

  let filePath = safeResolve(ROOT, pathname);
  if (!filePath) {
    return send(res, 403, "Forbidden");
  }

  if (pathname === "/") {
    filePath = path.join(ROOT, "demo/session_mode/index.html");
  }

  filePath = maybeDirectoryIndex(filePath);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === "ENOENT") {
        return send(res, 404, `Not found: ${pathname}`);
      }
      return send(res, 500, `Server error: ${err.message}`);
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME[ext] || "application/octet-stream";
    res.writeHead(200, {
      "Content-Type": contentType,
      "Cache-Control": "no-store",
    });
    res.end(data);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`Static server running at http://${HOST}:${PORT}`);
  console.log(`Root: ${ROOT}`);
  console.log(`Session demo: http://${HOST}:${PORT}/demo/session_mode/index.html`);
  console.log(`Browser client: http://${HOST}:${PORT}/demo/browser-client/index.html`);
});