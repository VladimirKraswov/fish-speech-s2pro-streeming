#!/usr/bin/env node

const fs = require("fs");
const fsp = require("fs/promises");
const http = require("http");
const path = require("path");
const { spawnSync } = require("child_process");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

const SCENARIOS = {
  short_a: "Привет. Это короткая проверка качества речи.",
  short_b: "Нам нужен ровный старт без щелчка и без обрывка слова.",
  short_c: "Сейчас важно, чтобы дикция оставалась чистой даже в стриме.",
  long_a:
    "Сегодня мы проверяем потоковую озвучку. Текст должен делиться только на безопасных границах, чтобы слова не рубились. Каждый аудиочанк сохраняется отдельно, а затем мы сравниваем склеенный итог.",
  long_b:
    "Сначала приходит несколько коротких дельт текста, затем появляется более длинная фраза без резких пауз, и система должна спокойно дождаться хорошей границы для коммита. После этого мы добавляем ещё пару предложений, чтобы проверить очередь TTS и отсутствие дыр между аудиофрагментами.",
  long_c:
    "Если браузерный playback-path всё ещё портит речь, это должно быть видно по расхождению между raw incoming PCM и пост-ресемпленным сигналом. В идеале обе версии звучат близко, а автоматическая ASR-проверка подтверждает это не только на слух.",
};

const VARIANTS = {
  final: {
    forceAudioContextSampleRate: 44100,
    resampleMode: "cubic",
    prebufferMs: 180,
  },
  token_batches: {
    forceAudioContextSampleRate: 44100,
    resampleMode: "cubic",
    prebufferMs: 220,
    streamTokens: false,
  },
  token_batches_260: {
    forceAudioContextSampleRate: 44100,
    resampleMode: "cubic",
    prebufferMs: 260,
    streamTokens: false,
  },
  system_cubic: {
    forceAudioContextSampleRate: null,
    resampleMode: "cubic",
    prebufferMs: 180,
  },
  system_linear: {
    forceAudioContextSampleRate: null,
    resampleMode: "linear",
    prebufferMs: 180,
  },
  forced_48000_linear: {
    forceAudioContextSampleRate: 48000,
    resampleMode: "linear",
    prebufferMs: 180,
  },
  forced_48000_cubic: {
    forceAudioContextSampleRate: 48000,
    resampleMode: "cubic",
    prebufferMs: 180,
  },
  chunk_24_24: {
    forceAudioContextSampleRate: 44100,
    resampleMode: "cubic",
    prebufferMs: 220,
    streamChunkSize: 24,
    initialStreamChunkSize: 24,
  },
  chunk_16_24: {
    forceAudioContextSampleRate: 44100,
    resampleMode: "cubic",
    prebufferMs: 200,
    streamChunkSize: 16,
    initialStreamChunkSize: 24,
  },
};

const DEFAULTS = {
  wsUrl: "ws://192.168.31.32:8765/ws",
  ttsBaseUrl: "http://192.168.31.32:8080",
  pagePath: "/demo/session_mode/index.html",
  scenario: "all",
  variant: "final",
  outDir: path.join(ROOT, "results", "session_quality"),
  timeoutSec: 240,
  settleMs: 1500,
  asrPython: process.env.FISH_ASR_PYTHON || "/tmp/fish-audio-venv/bin/python",
};

function parseArgs(argv) {
  const args = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const next = argv[i + 1];
    switch (token) {
      case "--ws-url":
        args.wsUrl = next;
        i += 1;
        break;
      case "--tts-base-url":
        args.ttsBaseUrl = next;
        i += 1;
        break;
      case "--scenario":
        args.scenario = next;
        i += 1;
        break;
      case "--variant":
        args.variant = next;
        i += 1;
        break;
      case "--out-dir":
        args.outDir = path.resolve(next);
        i += 1;
        break;
      case "--timeout-sec":
        args.timeoutSec = Number(next);
        i += 1;
        break;
      case "--asr-python":
        args.asrPython = next;
        i += 1;
        break;
      default:
        break;
    }
  }
  if (!VARIANTS[args.variant]) {
    throw new Error(`Unknown variant: ${args.variant}`);
  }
  if (args.scenario !== "all" && !SCENARIOS[args.scenario]) {
    throw new Error(`Unknown scenario: ${args.scenario}`);
  }
  return args;
}

function timestampDir() {
  const date = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_` +
    `${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
  );
}

function startStaticServer(rootDir) {
  const server = http.createServer(async (req, res) => {
    try {
      const pathname = decodeURIComponent(new URL(req.url, "http://127.0.0.1").pathname);
      const relativePath = pathname === "/" ? DEFAULTS.pagePath : pathname;
      const target = path.resolve(rootDir, `.${relativePath}`);
      if (!target.startsWith(rootDir)) {
        res.writeHead(403).end("forbidden");
        return;
      }
      const stat = await fsp.stat(target);
      if (stat.isDirectory()) {
        res.writeHead(404).end("not found");
        return;
      }
      const body = await fsp.readFile(target);
      res.writeHead(200, {
        "content-type": MIME_TYPES[path.extname(target)] || "application/octet-stream",
      });
      res.end(body);
    } catch (error) {
      res.writeHead(404).end(`not found: ${error.message}`);
    }
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve({
        server,
        baseUrl: `http://127.0.0.1:${address.port}`,
      });
    });
  });
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function decodeBase64Bytes(value) {
  return Buffer.from(value, "base64");
}

function decodeBase64Float32(value) {
  const buffer = decodeBase64Bytes(value);
  return new Float32Array(buffer.buffer, buffer.byteOffset, Math.floor(buffer.byteLength / 4)).slice();
}

function concatBuffers(buffers) {
  return Buffer.concat(buffers);
}

function concatFloat32(arrays) {
  let total = 0;
  for (const array of arrays) total += array.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const array of arrays) {
    out.set(array, offset);
    offset += array.length;
  }
  return out;
}

function writePcm16Wav(targetPath, pcmBytes, sampleRate, channels) {
  const byteRate = sampleRate * channels * 2;
  const blockAlign = channels * 2;
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcmBytes.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcmBytes.length, 40);
  fs.writeFileSync(targetPath, Buffer.concat([header, pcmBytes]));
}

function writeFloat32MonoWav(targetPath, samples, sampleRate) {
  const pcm = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    pcm.writeInt16LE(Math.round(clamped * 32767), i * 2);
  }
  writePcm16Wav(targetPath, pcm, sampleRate, 1);
}

function pcm16ToMonoFloat32(buffer, channels) {
  const frames = Math.floor(buffer.length / Math.max(1, channels) / 2);
  const out = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let acc = 0;
    for (let ch = 0; ch < channels; ch += 1) {
      acc += buffer.readInt16LE((frame * channels + ch) * 2) / 32768;
    }
    out[frame] = acc / Math.max(1, channels);
  }
  return out;
}

function floatMetrics(samples) {
  if (!samples.length) {
    return {
      peak: 0,
      rms: 0,
      dcOffset: 0,
      clippingCount: 0,
      clippingRatio: 0,
      hissScore: 0,
    };
  }
  let peak = 0;
  let sumSq = 0;
  let sum = 0;
  let clippingCount = 0;
  let diffSq = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const value = samples[i];
    const abs = Math.abs(value);
    if (abs > peak) peak = abs;
    if (abs >= 0.999) clippingCount += 1;
    sumSq += value * value;
    sum += value;
    if (i > 0) {
      const diff = value - samples[i - 1];
      diffSq += diff * diff;
    }
  }
  const rms = Math.sqrt(sumSq / samples.length);
  const dcOffset = sum / samples.length;
  const derivativeRms = samples.length > 1 ? Math.sqrt(diffSq / (samples.length - 1)) : 0;
  return {
    peak,
    rms,
    dcOffset,
    clippingCount,
    clippingRatio: clippingCount / samples.length,
    hissScore: rms > 1e-9 ? derivativeRms / rms : 0,
  };
}

function summarizeSeries(values) {
  if (!values.length) {
    return { min: null, avg: null, max: null };
  }
  const total = values.reduce((acc, value) => acc + value, 0);
  return {
    min: Number(Math.min(...values).toFixed(3)),
    avg: Number((total / values.length).toFixed(3)),
    max: Number(Math.max(...values).toFixed(3)),
  };
}

function summarizeScenarioMetrics(dump, rawFloat, resampledFloat) {
  const queueMs = dump.playback.queueDepthMsHistory.map((item) => Number(item.ms || 0));
  const discontinuityThreshold = 0.12;
  const boundaryHits = dump.playback.boundaryMetrics.filter(
    (item) => Number(item.discontinuity || 0) >= discontinuityThreshold,
  ).length;

  return {
    ttfaMs: dump.ttfaMs,
    totalPlaybackDurationMs: Number(
      ((resampledFloat.length / Math.max(1, dump.audioContextSampleRate || dump.audioMeta.sampleRate)) * 1000).toFixed(3),
    ),
    queueDepthMs: summarizeSeries(queueMs),
    underruns: dump.playback.underruns,
    rebufferCount: dump.playback.rebufferCount,
    gapCount: dump.playback.gapCount,
    overlapCount: dump.playback.overlapCount,
    discontinuityThreshold,
    discontinuityCount: boundaryHits,
    rawMetrics: floatMetrics(rawFloat),
    postResampleMetrics: floatMetrics(resampledFloat),
    interArrivalMs: summarizeSeries(dump.playback.interArrivalMs || []),
  };
}

function runAsr(asrPython, wavPath, expectedText) {
  if (!asrPython || !fs.existsSync(asrPython)) {
    return null;
  }
  const result = spawnSync(
    asrPython,
    [path.join(ROOT, "scripts", "asr_compare.py"), wavPath, expectedText],
    {
      cwd: ROOT,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
      env: {
        ...process.env,
        KMP_WARNINGS: "0",
      },
    },
  );
  if (result.status !== 0) {
    return {
      error: result.stderr || result.stdout || `exit_status=${result.status}`,
    };
  }
  try {
    const trimmed = result.stdout.trim();
    const start = trimmed.lastIndexOf("\n{");
    const jsonText = start >= 0 ? trimmed.slice(start + 1) : trimmed;
    return JSON.parse(jsonText);
  } catch (error) {
    return { error: error.message, raw: result.stdout };
  }
}

async function runScenario(page, baseUrl, args, scenarioName, text) {
  await page.goto(`${baseUrl}${args.pagePath}`, { waitUntil: "load" });
  await page.evaluate((variantConfig) => {
    window.__sessionDemoTestHooks.set(variantConfig);
  }, VARIANTS[args.variant]);

  await page.fill("#wsUrl", args.wsUrl);
  await page.fill("#ttsBaseUrl", args.ttsBaseUrl);
  await page.fill("#sourceText", text);
  await page.click("#connectBtn");
  await page.waitForFunction(() => window.__sessionDemoIsConnected && window.__sessionDemoIsConnected(), null, {
    timeout: args.timeoutSec * 1000,
  });
  await page.click("#startMockBtn");
  await page.waitForFunction(() => window.__sessionDemoIsIdle && window.__sessionDemoIsIdle(), null, {
    timeout: args.timeoutSec * 1000,
  });
  await page.waitForTimeout(args.settleMs);
  const dump = await page.evaluate(() => window.__sessionDemoDebugDump());
  await page.click("#disconnectBtn");
  await page.waitForTimeout(500);
  return { scenarioName, text, dump };
}

async function writeScenarioArtifacts(scenarioDir, scenarioName, text, dump, args) {
  ensureDir(scenarioDir);
  const rawFrames = dump.playback.rawFrames.map((frame) => ({
    ...frame,
    bytes: decodeBase64Bytes(frame.bytesB64),
  }));
  const resampledFrames = dump.playback.postResampleFrames.map((frame) => ({
    ...frame,
    samples: decodeBase64Float32(frame.float32B64),
  }));

  const joinedRawBytes = concatBuffers(rawFrames.map((frame) => frame.bytes));
  const joinedRawFloat = pcm16ToMonoFloat32(joinedRawBytes, dump.audioMeta.channels || 1);
  const joinedResampledFloat = concatFloat32(resampledFrames.map((frame) => frame.samples));

  const rawJoinedPath = path.join(scenarioDir, "browser_raw_joined.wav");
  const rawTransportPath = path.join(scenarioDir, "browser_transport_joined.wav");
  const postResampledPath = path.join(scenarioDir, "browser_post_resample.wav");

  writePcm16Wav(rawJoinedPath, joinedRawBytes, dump.audioMeta.sampleRate, dump.audioMeta.channels);
  writePcm16Wav(rawTransportPath, joinedRawBytes, dump.audioMeta.sampleRate, dump.audioMeta.channels);
  writeFloat32MonoWav(postResampledPath, joinedResampledFloat, dump.audioContextSampleRate || dump.audioMeta.sampleRate);

  const rawChunksDir = path.join(scenarioDir, "browser_raw_chunks");
  const resampledChunksDir = path.join(scenarioDir, "browser_post_resample_chunks");
  ensureDir(rawChunksDir);
  ensureDir(resampledChunksDir);

  for (const entry of dump.playback.rawPerChunk) {
    const buffers = entry.chunks.map(decodeBase64Bytes);
    writePcm16Wav(
      path.join(rawChunksDir, `${entry.chunkId}.wav`),
      concatBuffers(buffers),
      entry.sampleRate,
      entry.channels,
    );
  }

  for (const entry of dump.playback.postResamplePerChunk) {
    const chunks = entry.chunks.map(decodeBase64Float32);
    writeFloat32MonoWav(
      path.join(resampledChunksDir, `${entry.chunkId}.wav`),
      concatFloat32(chunks),
      entry.sampleRate,
    );
  }

  const metrics = summarizeScenarioMetrics(dump, joinedRawFloat, joinedResampledFloat);
  const asrRaw = runAsr(args.asrPython, rawJoinedPath, text);
  const asrPost = runAsr(args.asrPython, postResampledPath, text);

  const result = {
    scenario: scenarioName,
    text,
    variant: args.variant,
    ttfaMs: metrics.ttfaMs,
    totalPlaybackDurationMs: metrics.totalPlaybackDurationMs,
    queueDepthMs: metrics.queueDepthMs,
    underruns: metrics.underruns,
    rebufferCount: metrics.rebufferCount,
    gapCount: metrics.gapCount,
    overlapCount: metrics.overlapCount,
    discontinuityThreshold: metrics.discontinuityThreshold,
    discontinuityCount: metrics.discontinuityCount,
    interArrivalMs: metrics.interArrivalMs,
    rawMetrics: metrics.rawMetrics,
    postResampleMetrics: metrics.postResampleMetrics,
    audioContextSampleRate: dump.audioContextSampleRate,
    serverSampleRate: dump.audioMeta.sampleRate,
    serverChannels: dump.audioMeta.channels,
    serverDebugWavs: dump.playback.serverDebugWavs,
    joinedServerDebugWavPath: dump.playback.joinedServerDebugWavPath,
    rawJoinedPath,
    rawTransportPath,
    postResampledPath,
    asrRaw,
    asrPost,
    logLinesPath: path.join(scenarioDir, "browser_log.txt"),
  };

  await fsp.writeFile(
    result.logLinesPath,
    dump.logLines.join("\n") + "\n",
    "utf-8",
  );
  await fsp.writeFile(
    path.join(scenarioDir, "diagnostics.json"),
    JSON.stringify(dump, null, 2),
    "utf-8",
  );
  await fsp.writeFile(
    path.join(scenarioDir, "summary.json"),
    JSON.stringify(result, null, 2),
    "utf-8",
  );

  return result;
}

function writeMarkdownReport(outRoot, args, results) {
  const lines = [
    "# Browser Session Capture",
    "",
    `- variant: \`${args.variant}\``,
    `- ws_url: \`${args.wsUrl}\``,
    `- tts_base_url: \`${args.ttsBaseUrl}\``,
    `- audio_context_target_rate: \`${String(VARIANTS[args.variant].forceAudioContextSampleRate)}\``,
    "",
    "| scenario | ttfa_ms | duration_ms | underruns | rebuffer | gaps | queue_min_ms | queue_avg_ms | queue_max_ms | disc_count | raw_wer | post_wer | raw_joined | post_resample |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
  ];

  for (const item of results) {
    lines.push(
      `| ${item.scenario} | ${item.ttfaMs ?? "—"} | ${item.totalPlaybackDurationMs} | ${item.underruns} | ` +
      `${item.rebufferCount} | ${item.gapCount} | ${item.queueDepthMs.min ?? "—"} | ` +
      `${item.queueDepthMs.avg ?? "—"} | ${item.queueDepthMs.max ?? "—"} | ${item.discontinuityCount} | ` +
      `${item.asrRaw && item.asrRaw.wer != null ? item.asrRaw.wer.toFixed(3) : "—"} | ` +
      `${item.asrPost && item.asrPost.wer != null ? item.asrPost.wer.toFixed(3) : "—"} | ` +
      `${path.relative(outRoot, item.rawJoinedPath)} | ${path.relative(outRoot, item.postResampledPath)} |`,
    );
  }

  fs.writeFileSync(path.join(outRoot, "summary.md"), `${lines.join("\n")}\n`, "utf-8");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const scenarios = args.scenario === "all"
    ? Object.entries(SCENARIOS)
    : [[args.scenario, SCENARIOS[args.scenario]]];

  const timestamp = timestampDir();
  const outRoot = path.join(args.outDir, timestamp, args.variant);
  ensureDir(outRoot);

  const { server, baseUrl } = await startStaticServer(ROOT);
  const browser = await chromium.launch({
    headless: true,
    args: ["--autoplay-policy=no-user-gesture-required"],
  });

  const page = await browser.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") {
      process.stderr.write(`[browser:${message.type()}] ${message.text()}\n`);
    }
  });

  const results = [];
  try {
    for (const [scenarioName, text] of scenarios) {
      const scenarioDir = path.join(outRoot, scenarioName);
      const { dump } = await runScenario(page, baseUrl, args, scenarioName, text);
      const result = await writeScenarioArtifacts(scenarioDir, scenarioName, text, dump, args);
      results.push(result);
    }
  } finally {
    await browser.close();
    server.close();
  }

  await fsp.writeFile(
    path.join(outRoot, "summary.json"),
    JSON.stringify(results, null, 2),
    "utf-8",
  );
  writeMarkdownReport(outRoot, args, results);
  process.stdout.write(`${outRoot}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
