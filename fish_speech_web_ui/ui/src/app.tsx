import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { FishProxyClient } from './api/client';
import { AudioEngine } from './audio/engine';
import { LLMSimulator, type ChunkMode } from './lib/simulator';
import type { CommittedItem, ProxyConfig, StreamEvent } from './types';
import './app.css';

const HOST = window.location.hostname;
const PROXY_URL = `http://${HOST}:9000`;
const SERVER_URL = `http://${HOST}:8080`;

const DEFAULT_TEXT =
  'Встретил он по дороге Зайчика, Волка и Медведя, спел им свою песенку и убежал от них. Катился дальше Колобок и повстречалась ему Лисичка. Она и говорит: Колобок, Колобок, какая у тебя красивая песенка. А Колобок отвечает: я тебе её сейчас ещё раз спою.';

function makePresetConfig(kind: 'balanced' | 'lowLatency' | 'stable'): ProxyConfig {
  const base: ProxyConfig = {
    commit: {
      first: {
        min_chars: 18,
        target_chars: 42,
        max_chars: 90,
        max_wait_ms: 220,
        allow_partial_after_ms: 520,
      },
      next: {
        min_chars: 70,
        target_chars: 145,
        max_chars: 220,
        max_wait_ms: 650,
        allow_partial_after_ms: 1350,
      },
      flush_on_sentence_punctuation: true,
      flush_on_clause_punctuation: false,
      flush_on_newline: true,
      carry_incomplete_tail: true,
    },
    tts: {
      reference_id: 'voice',
      format: 'wav',
      normalize: true,
      use_memory_cache: 'on',
      seed: null,
      max_new_tokens: 300,
      chunk_length: 220,
      top_p: 0.82,
      repetition_penalty: 1.03,
      temperature: 0.7,
      stream_tokens: true,
      initial_stream_chunk_size: 28,
      stream_chunk_size: 20,
      first_initial_stream_chunk_size: 10,
      first_stream_chunk_size: 8,
      stateful_synthesis: true,
      stateful_fallback_to_stateless: false,
      stateful_history_turns: 1,
      stateful_history_chars: 160,
      stateful_history_code_frames: 120,
      stateful_reset_every_commits: 4,
      stateful_reset_every_chars: 800,
    },
    playback: {
      target_emit_bytes: 24576,
      start_buffer_ms: 300,
      first_commit_target_emit_bytes: 8192,
      first_commit_start_buffer_ms: 140,
      client_start_buffer_ms: 180,
      client_initial_start_delay_ms: 50,
      stop_grace_ms: 0,
      boundary_smoothing_enabled: true,
      punctuation_pauses_enabled: true,
      fade_in_ms: 8,
      fade_out_ms: 12,
      pause_after_clause_ms: 110,
      pause_after_sentence_ms: 280,
      pause_after_newline_ms: 520,
      pause_after_force_ms: 220,
      pause_after_hard_limit_ms: 40,
    },
    session: {
      max_buffer_chars: 20000,
      auto_close_on_finish: false,
    },
  };

  if (kind === 'lowLatency') {
    base.commit.first = {
      min_chars: 12,
      target_chars: 28,
      max_chars: 64,
      max_wait_ms: 140,
      allow_partial_after_ms: 320,
    };

    base.commit.next = {
      min_chars: 55,
      target_chars: 120,
      max_chars: 190,
      max_wait_ms: 480,
      allow_partial_after_ms: 950,
    };

    base.tts.max_new_tokens = 300;
    base.tts.chunk_length = 180;
    base.tts.initial_stream_chunk_size = 22;
    base.tts.stream_chunk_size = 16;
    base.tts.first_initial_stream_chunk_size = 6;
    base.tts.first_stream_chunk_size = 5;

    base.tts.stateful_history_turns = 1;
    base.tts.stateful_history_chars = 120;
    base.tts.stateful_history_code_frames = 96;
    base.tts.stateful_reset_every_commits = 3;
    base.tts.stateful_reset_every_chars = 650;

    base.playback.target_emit_bytes = 16384;
    base.playback.start_buffer_ms = 220;
    base.playback.first_commit_target_emit_bytes = 6144;
    base.playback.first_commit_start_buffer_ms = 90;
    base.playback.client_start_buffer_ms = 120;
    base.playback.client_initial_start_delay_ms = 25;
  }

  if (kind === 'stable') {
    base.commit.first = {
      min_chars: 24,
      target_chars: 56,
      max_chars: 110,
      max_wait_ms: 300,
      allow_partial_after_ms: 700,
    };

    base.commit.next = {
      min_chars: 90,
      target_chars: 180,
      max_chars: 250,
      max_wait_ms: 900,
      allow_partial_after_ms: 1800,
    };

    base.tts.max_new_tokens = 300;
    base.tts.chunk_length = 220;
    base.tts.initial_stream_chunk_size = 36;
    base.tts.stream_chunk_size = 28;
    base.tts.first_initial_stream_chunk_size = 14;
    base.tts.first_stream_chunk_size = 10;

    base.tts.stateful_history_turns = 1;
    base.tts.stateful_history_chars = 220;
    base.tts.stateful_history_code_frames = 160;
    base.tts.stateful_reset_every_commits = 5;
    base.tts.stateful_reset_every_chars = 1000;

    base.playback.target_emit_bytes = 32768;
    base.playback.start_buffer_ms = 650;
    base.playback.first_commit_target_emit_bytes = 12288;
    base.playback.first_commit_start_buffer_ms = 180;
    base.playback.client_start_buffer_ms = 240;
    base.playback.client_initial_start_delay_ms = 70;
  }

  return base;
}

const PRESETS = {
  balanced: {
    title: 'Balanced',
    config: makePresetConfig('balanced'),
  },
  lowLatency: {
    title: 'Low latency',
    config: makePresetConfig('lowLatency'),
  },
  stable: {
    title: 'Stable',
    config: makePresetConfig('stable'),
  },
};

type Health = any;

function jsonPretty(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function StatusPill({ label, health }: { label: string; health: Health }) {
  const ok = health && (health.ok || health.status === 'ok');
  const loading = health === null;

  return (
    <div class={`status-pill ${ok ? 'ok' : loading ? 'loading' : 'bad'}`}>
      <span class="dot" />
      <span>{label}</span>
      <b>{loading ? 'checking' : ok ? 'online' : 'offline'}</b>
    </div>
  );
}

export function App() {
  const [proxyHealth, setProxyHealth] = useState<Health>(null);
  const [serverHealth, setServerHealth] = useState<Health>(null);
  const [webUiHealth, setWebUiHealth] = useState<Health>(null);

  const [preset, setPreset] = useState<keyof typeof PRESETS>('balanced');
  const [configText, setConfigText] = useState(jsonPretty(PRESETS.balanced.config));
  const [text, setText] = useState(DEFAULT_TEXT);

  const [mode, setMode] = useState<ChunkMode>('words');
  const [minSize] = useState(3);
  const [maxSize] = useState(8);
  const [intervalMs, setIntervalMs] = useState(100);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState('idle');
  const [audioStatus, setAudioStatus] = useState('idle');
  const [isStreaming, setIsStreaming] = useState(false);

  const [committed, setCommitted] = useState<CommittedItem[]>([]);
  const [activeCommit, setActiveCommit] = useState('');
  const [logs, setLogs] = useState<string[]>([]);

  const client = useRef(new FishProxyClient(PROXY_URL));
  const abortController = useRef<AbortController | null>(null);
  const simulator = useRef<LLMSimulator | null>(null);

  const [speedMultiplier, setSpeedMultiplier] = useState(1.0);

  const log = (message: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev.slice(-180), `[${time}] ${message}`]);
  };

  const onStreamEvent = (event: StreamEvent) => {
    if (event.type === 'session_start') {
      log(
        `pcm stream opened, target_emit_bytes=${event.target_emit_bytes ?? 'n/a'}, ` +
        `first_emit=${event.first_commit_target_emit_bytes ?? 'n/a'}, ` +
        `client_buffer=${event.client_start_buffer_ms ?? 'n/a'}ms`
      );
    }

    if (event.type === 'upstream_reset') {
      log(`upstream synthesis reset at commit #${event.commit_seq}, reason=${event.reason}`);
    }

    if (event.type === 'upstream_reset_failed') {
      log(`upstream synthesis reset FAILED at commit #${event.commit_seq}, reason=${event.reason}: ${event.message}`);
    }

    if (event.type === 'commit_start') {
      setActiveCommit(event.text_preview || event.text || '');
      log(
        `commit #${event.commit_seq} started, reason=${event.reason}, ` +
        `emit=${event.effective_target_emit_bytes ?? 'n/a'}, ` +
        `start_buffer=${event.effective_start_buffer_ms ?? 'n/a'}ms`
      );
    }

    if (event.type === 'pcm' && event.commit_seq === 1 && event.first_pcm_for_commit) {
      log('first PCM for commit #1 received');
    }

    if (event.type === 'pause') {
      log(`pause #${event.commit_seq}: ${event.pause_ms}ms, boundary=${event.boundary}`);
    }

    if (event.type === 'commit_done') {
      setActiveCommit('');
      log(
        `commit #${event.commit_seq} done, upstream_bytes=${event.upstream_bytes ?? 'n/a'}, ` +
          `pause=${event.pause_ms ?? 0}ms, boundary=${event.boundary ?? 'n/a'}`,
      );
    }

    if (event.type === 'error') {
      log(`stream error: ${event.message}`);
      setIsStreaming(false);
    }

    if (event.type === 'session_done') {
      setSessionStatus('finished');
      setIsStreaming(false);
      log(`session done, commits=${event.commit_count ?? 'n/a'}`);
    }

    if (event.type === 'session_aborted') {
      setSessionStatus('aborted');
      setIsStreaming(false);
      log('session aborted');
    }
  };

  const audioEngine = useRef(new AudioEngine(setAudioStatus, onStreamEvent));

  const configError = useMemo(() => {
    try {
      JSON.parse(configText);
      return '';
    } catch (error) {
      return error instanceof Error ? error.message : 'Invalid JSON';
    }
  }, [configText]);

  const refreshHealth = async () => {
    try {
      setProxyHealth(await client.current.getHealth());
    } catch {
      setProxyHealth({ ok: false });
    }

    try {
      setServerHealth(await fetch(`${SERVER_URL}/v1/health`, { cache: 'no-store' }).then((r) => r.json()));
    } catch {
      setServerHealth({ ok: false });
    }

    try {
      setWebUiHealth(await fetch('/health', { cache: 'no-store' }).then((r) => r.json()));
    } catch {
      setWebUiHealth({ ok: false });
    }
  };

  useEffect(() => {
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const applyPreset = (value: keyof typeof PRESETS) => {
    setPreset(value);
    setConfigText(jsonPretty(PRESETS[value].config));
    log(`preset selected: ${PRESETS[value].title}`);
  };

  const addCommitted = (items: CommittedItem[] | undefined) => {
    if (!items?.length) return;
    setCommitted((prev) => [...items, ...prev].slice(0, 100));
  };

  const openSession = async () => {
    if (configError) {
      log(`config error: ${configError}`);
      return;
    }

    simulator.current?.stop();
    abortController.current?.abort();
    audioEngine.current.reset();

    setSessionStatus('opening');
    setCommitted([]);
    setActiveCommit('');

    try {
      const data = await client.current.openSession(configText);

      setSessionId(data.session_id);
      setSessionStatus('open');

      const stateful = data.config?.tts?.stateful_synthesis ? 'stateful' : 'stateless';
      const maxTokens = data.config?.tts?.max_new_tokens ?? 'n/a';
      const chunkLength = data.config?.tts?.chunk_length ?? 'n/a';
      const historyTurns = data.config?.tts?.stateful_history_turns ?? 'n/a';
      const historyFrames = data.config?.tts?.stateful_history_code_frames ?? 'n/a';

      log(
        `session opened: ${data.session_id.slice(0, 8)}, mode=${stateful}, max_new_tokens=${maxTokens}, chunk_length=${chunkLength}, history_turns=${historyTurns}, history_frames=${historyFrames}`,
      );

      abortController.current = new AbortController();

      audioEngine.current
        .connectStream(client.current.getStreamUrl(data.session_id), abortController.current.signal)
        .catch((error) => {
          if (error?.name !== 'AbortError') {
            log(`audio error: ${error instanceof Error ? error.message : String(error)}`);
          }
        });
    } catch (error) {
      setSessionStatus('idle');
      log(`open failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const finishSession = async () => {
    if (!sessionId) return;

    setSessionStatus('finishing');

    try {
      const data = await client.current.finishSession(sessionId, 'llm_input_finished');
      addCommitted(data.committed);

      if (data.committed?.length) {
        log(`finish committed ${data.committed.length} segment(s)`);
      } else {
        log('finish sent, no buffered text');
      }
    } catch (error) {
      setSessionStatus('open');
      log(`finish failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const closeSession = async () => {
    if (!sessionId) return;

    const id = sessionId;

    simulator.current?.stop();
    abortController.current?.abort();
    audioEngine.current.reset();

    setIsStreaming(false);
    setSessionStatus('closing');

    try {
      await client.current.closeSession(id);
      log(`session closed: ${id.slice(0, 8)}`);
    } catch (error) {
      log(`close failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    setSessionId(null);
    setSessionStatus('idle');
    setCommitted([]);
    setActiveCommit('');
  };

  const startTextStream = () => {
    if (!sessionId || isStreaming) return;

    setIsStreaming(true);
    setSessionStatus('open');
    log('LLM text simulation started');

    simulator.current = new LLMSimulator({
      mode,
      minSize,
      maxSize,
      interval: intervalMs,
      speedMultiplier,
      onChunk: async (chunk) => {
        try {
          const data = await client.current.appendText(sessionId, chunk);
          addCommitted(data.committed);
          return true;
        } catch (error) {
          log(`append failed: ${error instanceof Error ? error.message : String(error)}`);
          setIsStreaming(false);
          return false;
        }
      },
      onFinish: async () => {
        setIsStreaming(false);
        log('LLM text simulation finished');
        await finishSession();
      },
    });

    simulator.current.start(text);
  };

  const stopTextStream = async () => {
    simulator.current?.stop();
    setIsStreaming(false);
    log('LLM text simulation stopped');
    await finishSession();
  };

  const flush = async () => {
    if (!sessionId) return;

    try {
      const data = await client.current.flushSession(sessionId, 'manual_flush');
      addCommitted(data.committed);
      log(`manual flush, committed=${data.committed?.length ?? 0}`);
    } catch (error) {
      log(`flush failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  return (
    <main class="app-shell">
      <section class="hero">
        <div>
          <div class="eyebrow">Fish Speech S2 Pro Streaming</div>
          <h1>Reliable realtime voice console</h1>
          <p>
            Имитация LLM-вывода: текст отправляется маленькими чанками в proxy,
            а proxy собирает их в более естественные речевые сегменты с acoustic continuation.
          </p>
        </div>

        <div class="hero-actions">
          <button class="ghost" onClick={refreshHealth}>Refresh</button>

          {!sessionId ? (
            <button class="primary" onClick={openSession} disabled={sessionStatus === 'opening' || !!configError}>
              Open session
            </button>
          ) : (
            <button class="danger" onClick={closeSession}>
              Close session
            </button>
          )}
        </div>
      </section>

      <section class="status-grid">
        <StatusPill label="Server" health={serverHealth} />
        <StatusPill label="Proxy" health={proxyHealth} />
        <StatusPill label="Web UI" health={webUiHealth} />

        <div class="status-pill audio">
          <span class="dot" />
          <span>Audio</span>
          <b>{audioStatus}</b>
        </div>
      </section>

      <section class="main-grid">
        <div class="panel composer">
          <div class="panel-head">
            <div>
              <h2>LLM text stream</h2>
              <p>Вставь длинный текст и запусти постепенную отправку как будто это output LLM.</p>
            </div>
            <span class={`session-badge ${sessionId ? 'on' : ''}`}>{sessionStatus}</span>
          </div>

          <textarea
            class="text-input"
            value={text}
            onInput={(e) => setText((e.currentTarget as HTMLTextAreaElement).value)}
          />

          <div class="controls-grid">
            <label>
              Режим эмуляции
              <select value={mode} onChange={(e) => setMode((e.currentTarget as HTMLSelectElement).value as ChunkMode)}>
                <option value="tokens">Tokens (самый реалистичный)</option>
                <option value="words">По словам</option>
                <option value="chars">По символам</option>
              </select>
            </label>

            <label>
              Интервал (мс)
              <input
                type="number"
                value={intervalMs}
                min="10"
                step="10"
                onInput={(e) => setIntervalMs(Number((e.currentTarget as HTMLInputElement).value))}
              />
            </label>

            <label>
              Скорость
              <select 
                value={speedMultiplier} 
                onChange={(e) => setSpeedMultiplier(Number((e.currentTarget as HTMLSelectElement).value))}
              >
                <option value={0.6}>Быстро</option>
                <option value={1.0}>Нормально</option>
                <option value={1.4}>Медленно</option>
              </select>
            </label>
          </div>

          <div class="button-row">
            <button class="primary" onClick={startTextStream} disabled={!sessionId || isStreaming}>
              Start streaming
            </button>
            <button onClick={stopTextStream} disabled={!sessionId || !isStreaming}>Stop + finish</button>
            <button onClick={flush} disabled={!sessionId}>Force flush</button>
            <button onClick={finishSession} disabled={!sessionId || sessionStatus === 'finishing'}>Finish input</button>
          </div>
        </div>

        <div class="panel config-panel">
          <div class="panel-head">
            <div>
              <h2>Runtime preset</h2>
              <p>
                JSON override для `/session/open`. Balanced — основной профиль для плавной речи,
                Low latency — быстрее, Stable — самые мягкие переходы.
              </p>
            </div>
          </div>

          <div class="preset-row">
            {Object.entries(PRESETS).map(([key, value]) => (
              <button
                key={key}
                class={preset === key ? 'selected' : ''}
                onClick={() => applyPreset(key as keyof typeof PRESETS)}
                disabled={!!sessionId}
              >
                {value.title}
              </button>
            ))}
          </div>

          {configError && <div class="error-box">{configError}</div>}

          <textarea
            class="config-input"
            value={configText}
            spellcheck={false}
            onInput={(e) => setConfigText((e.currentTarget as HTMLTextAreaElement).value)}
            disabled={!!sessionId}
          />
        </div>
      </section>

      <section class="main-grid lower">
        <div class="panel">
          <div class="panel-head">
            <div>
              <h2>Committed segments</h2>
              <p>Сегменты, которые proxy уже отправил в TTS.</p>
            </div>
            <b>{committed.length}</b>
          </div>

          {activeCommit && (
            <div class="active-commit">
              <span>Now generating</span>
              <p>{activeCommit}</p>
            </div>
          )}

          <div class="timeline">
            {committed.length === 0 && <div class="empty">Пока нет committed сегментов.</div>}

            {committed.map((item) => (
              <article class="timeline-item" key={item.seq}>
                <div class="timeline-dot">#{item.seq}</div>
                <div>
                  <div class="timeline-meta">{item.reason}</div>
                  <p>{item.text}</p>
                </div>
              </article>
            ))}
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div>
              <h2>Event log</h2>
              <p>События UI, proxy stream и audio scheduler.</p>
            </div>
          </div>

          <div class="log-box">
            {logs.map((item, index) => (
              <div key={index}>{item}</div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
