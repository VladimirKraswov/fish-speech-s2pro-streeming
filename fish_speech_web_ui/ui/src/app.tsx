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
        min_chars: 55,
        target_chars: 110,
        max_chars: 180,
        max_wait_ms: 220,
        allow_partial_after_ms: 420,
      },
      next: {
        min_chars: 90,
        target_chars: 170,
        max_chars: 260,
        max_wait_ms: 420,
        allow_partial_after_ms: 800,
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
      max_new_tokens: 420,
      chunk_length: 160,
      top_p: 0.8,
      repetition_penalty: 1.06,
      temperature: 0.7,
      stream_tokens: true,
      initial_stream_chunk_size: 10,
      stream_chunk_size: 8,
      stateful_synthesis: false,
      stateful_fallback_to_stateless: true,
    },
    playback: {
      target_emit_bytes: 8192,
      start_buffer_ms: 240,
      stop_grace_ms: 250,
    },
    session: {
      max_buffer_chars: 20000,
      auto_close_on_finish: false,
    },
  };

  if (kind === 'lowLatency') {
    base.commit.first = {
      min_chars: 45,
      target_chars: 90,
      max_chars: 150,
      max_wait_ms: 160,
      allow_partial_after_ms: 320,
    };

    base.commit.next = {
      min_chars: 75,
      target_chars: 140,
      max_chars: 220,
      max_wait_ms: 320,
      allow_partial_after_ms: 650,
    };

    base.tts.max_new_tokens = 360;
    base.tts.chunk_length = 150;
    base.tts.initial_stream_chunk_size = 8;
    base.tts.stream_chunk_size = 6;

    base.playback.target_emit_bytes = 6144;
    base.playback.start_buffer_ms = 160;
    base.playback.stop_grace_ms = 180;
  }

  if (kind === 'stable') {
    base.commit.first = {
      min_chars: 70,
      target_chars: 140,
      max_chars: 220,
      max_wait_ms: 300,
      allow_partial_after_ms: 600,
    };

    base.commit.next = {
      min_chars: 120,
      target_chars: 220,
      max_chars: 340,
      max_wait_ms: 600,
      allow_partial_after_ms: 1100,
    };

    base.tts.max_new_tokens = 480;
    base.tts.chunk_length = 180;
    base.tts.initial_stream_chunk_size = 12;
    base.tts.stream_chunk_size = 10;

    base.playback.target_emit_bytes = 12288;
    base.playback.start_buffer_ms = 320;
    base.playback.stop_grace_ms = 350;
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
  const [minSize, setMinSize] = useState(3);
  const [maxSize, setMaxSize] = useState(8);
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

  const log = (message: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev.slice(-180), `[${time}] ${message}`]);
  };

  const onStreamEvent = (event: StreamEvent) => {
    if (event.type === 'session_start') {
      log(`pcm stream opened, target_emit_bytes=${event.target_emit_bytes ?? 'n/a'}`);
    }

    if (event.type === 'commit_start') {
      setActiveCommit(event.text || '');
      log(`commit #${event.commit_seq} started, reason=${event.reason}`);
    }

    if (event.type === 'commit_done') {
      setActiveCommit('');
      log(`commit #${event.commit_seq} done, upstream_bytes=${event.upstream_bytes ?? 'n/a'}`);
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

      log(`session opened: ${data.session_id.slice(0, 8)}, mode=${stateful}, max_new_tokens=${maxTokens}`);

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
            а proxy озвучивает его с увеличенным запасом генерации, чтобы не резать хвост фразы.
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
              Mode
              <select value={mode} onChange={(e) => setMode((e.currentTarget as HTMLSelectElement).value as ChunkMode)}>
                <option value="chars">Chars</option>
                <option value="words">Words</option>
                <option value="sentences">Sentences</option>
              </select>
            </label>

            <label>
              Min
              <input
                type="number"
                value={minSize}
                min="1"
                onInput={(e) => setMinSize(Number((e.currentTarget as HTMLInputElement).value))}
              />
            </label>

            <label>
              Max
              <input
                type="number"
                value={maxSize}
                min="1"
                onInput={(e) => setMaxSize(Number((e.currentTarget as HTMLInputElement).value))}
              />
            </label>

            <label>
              Interval ms
              <input
                type="number"
                value={intervalMs}
                min="10"
                onInput={(e) => setIntervalMs(Number((e.currentTarget as HTMLInputElement).value))}
              />
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
              <p>JSON override для `/session/open`. Balanced и Stable настроены на полное дочитывание.</p>
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