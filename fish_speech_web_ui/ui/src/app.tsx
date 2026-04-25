import { useState, useEffect, useRef } from 'preact/hooks';
import { Dashboard } from './components/Dashboard';
import { SessionManager } from './components/SessionManager';
import { Streamer } from './components/Streamer';
import { EventLog } from './components/EventLog';
import { FishProxyClient } from './api/client';
import { AudioEngine } from './audio/engine';
import { LLMSimulator } from './lib/simulator';
import type { StreamEvent, CommittedItem } from './types';
import './app.css';

const PROXY_URL = 'http://localhost:9000';
const SERVER_URL = 'http://localhost:8080';

const DEFAULT_CONFIG = {
    commit: {
        first: {
            min_chars: 40,
            target_chars: 58,
            max_chars: 84,
            max_wait_ms: 150,
            allow_partial_after_ms: 240
        },
        next: {
            min_chars: 120,
            target_chars: 160,
            max_chars: 240,
            max_wait_ms: 340,
            allow_partial_after_ms: 600
        },
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
    playback: {
        target_emit_bytes: 6144,
        start_buffer_ms: 120,
        stop_grace_ms: 60
    },
    session: {
        max_buffer_chars: 4000,
        auto_close_on_finish: false
    }
};

export function App() {
    const [proxyHealth, setProxyHealth] = useState<any>(null);
    const [serverHealth, setServerHealth] = useState<any>(null);
    const [webUiHealth, setWebUiHealth] = useState<any>(null);
    const [audioStatus, setAudioStatus] = useState('idle');
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [sessionStatus, setSessionStatus] = useState('idle');
    const [configText, setConfigText] = useState(JSON.stringify(DEFAULT_CONFIG, null, 2));
    const [logs, setLogs] = useState<string[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [committedItems, setCommittedItems] = useState<CommittedItem[]>([]);

    const client = useRef(new FishProxyClient(PROXY_URL));
    const audioEngine = useRef(new AudioEngine(setAudioStatus, (event) => handleStreamEvent(event)));
    const simulator = useRef<LLMSimulator | null>(null);
    const abortController = useRef<AbortController | null>(null);

    const log = (msg: string) => {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, `[${time}] ${msg}`]);
    };

    const handleStreamEvent = (event: StreamEvent) => {
        if (event.type === 'commit_start') {
            log(`TTS Commit Start: ${event.text.slice(0, 30)}...`);
        } else if (event.type === 'commit_done') {
            log(`TTS Commit Done: #${event.commit_seq}`);
        } else if (event.type === 'error') {
            log(`Stream error: ${event.message}`);
        }
    };

    const refreshHealth = async () => {
        try {
            const ph = await client.current.getHealth();
            setProxyHealth(ph);
        } catch { setProxyHealth({ ok: false }); }

        try {
            const sh = await fetch(`${SERVER_URL}/v1/health`).then(r => r.json());
            setServerHealth(sh);
        } catch { setServerHealth({ ok: false }); }

        try {
            const wh = await fetch('/health').then(r => r.json());
            setWebUiHealth(wh);
        } catch { setWebUiHealth({ ok: false }); }
    };

    useEffect(() => {
        refreshHealth();
        const timer = setInterval(refreshHealth, 5000);
        return () => clearInterval(timer);
    }, []);

    const onOpenSession = async () => {
        try {
            setSessionStatus('opening');
            log('Opening session...');
            const data = await client.current.openSession(configText);
            setSessionId(data.session_id);
            setSessionStatus('open');
            log(`Session opened: ${data.session_id}`);

            abortController.current = new AbortController();
            audioEngine.current.connectStream(
                client.current.getStreamUrl(data.session_id),
                abortController.current.signal
            ).catch(e => {
                if (e.name !== 'AbortError') log(`Audio stream error: ${e.message}`);
            });
        } catch (e: any) {
            setSessionStatus('idle');
            log(`Failed to open session: ${e.message}`);
        }
    };

    const onFinishSession = async () => {
        if (!sessionId) return;
        try {
            log('Finishing input...');
            await client.current.finishSession(sessionId);
            setSessionStatus('finishing');
        } catch (e: any) {
            log(`Failed to finish session: ${e.message}`);
        }
    };

    const onCloseSession = async () => {
        if (!sessionId) return;
        try {
            log('Closing session...');
            if (abortController.current) abortController.current.abort();
            audioEngine.current.reset();
            await client.current.closeSession(sessionId);
            setSessionId(null);
            setSessionStatus('idle');
            setCommittedItems([]);
            log('Session closed.');
        } catch (e: any) {
            log(`Failed to close session: ${e.message}`);
        }
    };

    const onStartStreaming = (text: string, options: any) => {
        if (!sessionId) return;
        setIsStreaming(true);
        log('Starting LLM simulation...');
        simulator.current = new LLMSimulator({
            ...options,
            onChunk: async (chunk: string) => {
                try {
                    const data = await client.current.appendText(sessionId, chunk);
                    if (data.committed && data.committed.length > 0) {
                        setCommittedItems(prev => [...data.committed, ...prev]);
                    }
                    return true;
                } catch (e: any) {
                    log(`Append error: ${e.message}`);
                    return false;
                }
            },
            onFinish: async () => {
                log('Simulation finished.');
                setIsStreaming(false);
                await onFinishSession();
            }
        });
        simulator.current.start(text);
    };

    const onStopStreaming = () => {
        if (simulator.current) {
            simulator.current.stop();
            setIsStreaming(false);
            log('Simulation stopped.');
        }
    };

    const onFlush = async () => {
        if (!sessionId) return;
        try {
            const data = await client.current.flushSession(sessionId);
            if (data.committed && data.committed.length > 0) {
                setCommittedItems(prev => [...data.committed, ...prev]);
            }
            log('Manual flush requested.');
        } catch (e: any) {
            log(`Flush error: ${e.message}`);
        }
    };

    return (
        <div class="container">
            <h1>Fish Speech — Session Stream</h1>
            <Dashboard
                proxyHealth={proxyHealth}
                serverHealth={serverHealth}
                webUiHealth={webUiHealth}
                audioStatus={audioStatus}
            />

            <div class="grid">
                <SessionManager
                    sessionId={sessionId}
                    status={sessionStatus}
                    configText={configText}
                    onConfigChange={setConfigText}
                    onOpen={onOpenSession}
                    onFinish={onFinishSession}
                    onClose={onCloseSession}
                />
                <Streamer
                    sessionOpen={sessionStatus === 'open'}
                    onStart={onStartStreaming}
                    onStop={onStopStreaming}
                    onFlush={onFlush}
                    isStreaming={isStreaming}
                />
            </div>

            <div class="grid">
                <div class="card">
                    <h3>3. Status Details</h3>
                    <div class="small">Session: {sessionStatus}</div>
                    <div class="small">Audio: {audioStatus}</div>
                    {sessionId && <div class="small mono">ID: {sessionId}</div>}
                </div>
                <div class="card">
                    <h3>4. Committed Segments</h3>
                    <div class="committed-list">
                        {committedItems.map(item => (
                            <div key={item.seq} class="chunk">
                                <div class="meta">#{item.seq} • {item.reason}</div>
                                <div>{item.text}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <EventLog logs={logs} />
        </div>
    );
}
