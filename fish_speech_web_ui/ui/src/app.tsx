import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { FishProxyClient } from './api/client';
import { AudioEngine } from './audio/engine';
import { LLMSimulator, type ChunkMode } from './lib/simulator';
import type { CommittedItem, StreamEvent } from './types';
import './app.css';

const PROXY_URL = `http://${window.location.hostname}:9000`;
const SERVER_URL = `http://${window.location.hostname}:8080`;

const DEFAULT_TEXT =
  'Встретил он по дороге Зайчика, Волка и Медведя, спел им свою песенку и убежал от них. Катился дальше Колобок и повстречалась ему Лисичка. Она и говорит: Колобок, Колобок, какая у тебя красивая песенка.';

const PRESETS = {
  balanced: {
    title: 'Balanced',
    config: {
      commit: {
        first: { min_chars: 40, target_chars: 58, max_chars: 84, max_wait_ms: 150, allow_partial_after_ms: 240 },
        next: { min_chars: 120, target_chars: 160, max_chars: 240, max_wait_ms: 340, allow_partial_after_ms: 600 },
        flush_on_sentence_punctuation: true,
        flush_on_clause_punctuation: true,
        flush_on_newline: true,
        carry_incomplete_tail: true
      },
      tts: {
        reference_id: 'voice',
        max_new_tokens: 160,
        chunk_length: 160,
        top_p: 0.78,
        repetition_penalty: 1.12,
        temperature: 0.7,
        initial_stream_chunk_size: 10,
        stream_chunk_size: 8
      },
      playback: { target_emit_bytes: 6144, start_buffer_ms: 120, stop_grace_ms: 60 },
      session: { max_buffer_chars: 4000, auto_close_on_finish: false }
    }
  },
  lowLatency: {
    title: 'Low latency',
    config: {
      commit: {
        first: { min_chars: 24, target_chars: 42, max_chars: 68, max_wait_ms: 100, allow_partial_after_ms: 170 },
        next: { min_chars: 70, target_chars: 110, max_chars: 170, max_wait_ms: 240, allow_partial_after_ms: 420 },
        flush_on_sentence_punctuation: true,
        flush_on_clause_punctuation: true,
        flush_on_newline: true,
        carry_incomplete_tail: true
      },
      tts: {
        reference_id: 'voice',
        max_new_tokens: 128,
        chunk_length: 140,
        top_p: 0.78,
        repetition_penalty: 1.12,
        temperature: 0.7,
        initial_stream_chunk_size: 8,
        stream_chunk_size: 6
      },
      playback: { target_emit_bytes: 4096, start_buffer_ms: 80, stop_grace_ms: 40 },
      session: { max_buffer_chars: 4000, auto_close_on_finish: false }
    }
  },
  stable: {
    title: 'Stable',
    config: {
      commit: {
        first: { min_chars: 60, target_chars: 90, max_chars: 130, max_wait_ms: 250, allow_partial_after_ms: 420 },
        next: { min_chars: 150, target_chars: 220, max_chars: 300, max_wait_ms: 520, allow_partial_after_ms: 900 },
        flush_on_sentence_punctuation: true,
        flush_on_clause_punctuation: true,
        flush_on_newline: true,
        carry_incomplete_tail: true
      },
      tts: {
        reference_id: 'voice',
        max_new_tokens: 180,
        chunk_length: 180,
        top_p: 0.78,
        repetition_penalty: 1.12,
        temperature: 0.7,
        initial_stream_chunk_size: 12,
        stream_chunk_size: 10
      },
      playback: { target_emit_bytes: 8192, start_buffer_ms: 180, stop_grace_ms: 80 },
      session: { max_buffer_chars: 4000, auto_close_on_finish: false }
    }
  }
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
  const [activeCommit, setActiveCommit] = useState<string>('');
  const [logs, setLogs] = useState<string[]>([]);

  const client = useRef(new FishProxyClient(PROXY_URL));
  const abortController = useRef<AbortController | null>(null);
  const simulator = useRef<LLMSimulator | null>(null);

  const log = (message: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev.slice(-160), `[${time}] ${message}`]);
  };

  const onStreamEvent = (event: StreamEvent) => {
    if (event.type === 'commit_start') {
      setActiveCommit(event.text || '');
      log(`commit #${event.commit_seq} started`);
    }
    if (event.type === 'commit_done') {
      setActiveCommit('');
      log(`commit #${event.commit_seq} done`);
    }
    if (event.type === 'error') {
      log(`stream error: ${event.message}`);
    }
    if (event.type === 'session_done') {
      setSessionStatus('finished');
      setIsStreaming(false);
      log('session done');
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
      setServerHealth(await fetch(`${SERVER_URL}/v1/health`).then((r) => r.json()));
    } catch {
      setServerHealth({ ok: false });
    }

    try {
      setWebUiHealth(await fetch('/health').then((r) => r.json()));
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

  const openSession = async () => {
    if (configError) {
      log(`config error: ${configError}`);
      return;
    }

    setSessionStatus('opening');
    setCommitted([]);
    setActiveCommit('');

    try {
      const data = await client.current.openSession(configText);
      setSessionId(data.session_id);
      setSessionStatus('open');
      log(`session opened: ${data.session_id.slice(0, 8)}`);

      abortController.current = new AbortController();
      audioEngine.current
        .connectStream(client.current.getStreamUrl(data.session_id), abortController.current.signal)
        .catch((error) => {
          if (error?.name !== 'AbortError') log(`audio error: ${error.message}`);
        });
    } catch (error) {
      setSessionStatus('idle');
      log(`open failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const finishSession = async () => {
    if (!sessionId) return;
    setSessionStatus('finishing');
    await client.current.finishSession(sessionId).catch((error) => log(`finish failed: ${error.message}`));
  };

  const closeSession = async () => {
    if (!sessionId) return;

    simulator.current?.stop();
    abortController.current?.abort();
    audioEngine.current.reset();

    await client.current.closeSession(sessionId).catch((error) => log(`close failed: ${error.message}`));

    setSessionId(null);
    setSessionStatus('idle');
    setIsStreaming(false);
    setCommitted([]);
    setActiveCommit('');
    log('session closed');
  };

  const startTextStream = () => {
    if (!sessionId || isStreaming) return;

    setIsStreaming(true);
    log('text stream started');

    simulator.current = new LLMSimulator({
      mode,
      minSize,
      maxSize,
      interval: intervalMs,
      onChunk: async (chunk) => {
        try {
          const data = await client.current.appendText(sessionId, chunk);
          if (data.committed?.length) {
            setCommitted((prev) => [...data.committed, ...prev].slice(0, 80));
          }
          return true;
        } catch (error) {
          log(`append failed: ${error instanceof Error ? error.message : String(error)}`);
          setIsStreaming(false);
          return false;
        }
      },
      onFinish: async () => {
        setIsStreaming(false);
        log('text stream finished');
        await finishSession();
      }
    });

    simulator.current.start(text);
  };

  const stopTextStream = () => {
    simulator.current?.stop();
    setIsStreaming(false);
    log('text stream stopped');
  };

  const flush = async () => {
    if (!sessionId) return;
    try {
      const data = await client.current.flushSession(sessionId);
      if (data.committed?.length) {
        setCommitted((prev) => [...data.committed, ...prev].slice(0, 80));
      }
      log('manual flush');
    } catch (error) {
      log(`flush failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  return (
    <main class="app-shell">
      <section class="hero">
        <div>
          <div class="eyebrow">Fish Speech S2 Pro Streaming</div>
          <h1>Realtime voice session console</h1>
          <p>Управление сессией, LLM-потоком, PCM playback и диагностикой в одном нормальном интерфейсе.</p>
        </div>

        <div class="hero-actions">
          <button class="ghost" onClick={refreshHealth}>Refresh</button>
          {!sessionId ? (
            <button class="primary" onClick={openSession} disabled={sessionStatus === 'opening' || !!configError}>
              Open session
            </button>
          ) : (
            <button class="danger" onClick={closeSession}>Close session</button>
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
              <h2>Text stream</h2>
              <p>Имитация входящего LLM текста с постепенной отправкой в proxy.</p>
            </div>
            <span class={`session-badge ${sessionId ? 'on' : ''}`}>{sessionStatus}</span>
          </div>

          <textarea class="text-input" value={text} onInput={(e) => setText((e.currentTarget as HTMLTextAreaElement).value)} />

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
              <input type="number" value={minSize} min="1" onInput={(e) => setMinSize(Number((e.currentTarget as HTMLInputElement).value))} />
            </label>

            <label>
              Max
              <input type="number" value={maxSize} min="1" onInput={(e) => setMaxSize(Number((e.currentTarget as HTMLInputElement).value))} />
            </label>

            <label>
              Interval ms
              <input type="number" value={intervalMs} min="10" onInput={(e) => setIntervalMs(Number((e.currentTarget as HTMLInputElement).value))} />
            </label>
          </div>

          <div class="button-row">
            <button class="primary" onClick={startTextStream} disabled={!sessionId || isStreaming}>
              Start streaming
            </button>
            <button onClick={stopTextStream} disabled={!isStreaming}>Stop</button>
            <button onClick={flush} disabled={!sessionId}>Force flush</button>
            <button onClick={finishSession} disabled={!sessionId || sessionStatus !== 'open'}>Finish input</button>
          </div>
        </div>

        <div class="panel config-panel">
          <div class="panel-head">
            <div>
              <h2>Runtime preset</h2>
              <p>JSON override для proxy session.</p>
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
              <p>Фразы, которые proxy уже отправил в TTS.</p>
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
              <p>Последние события UI/proxy/audio.</p>
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