import type { FunctionalComponent } from 'preact';
import { useState } from 'preact/hooks';
import type { ChunkMode } from '../lib/simulator';

interface StreamerProps {
    sessionOpen: boolean;
    onStart: (text: string, options: any) => void;
    onStop: () => void;
    onFlush: () => void;
    isStreaming: boolean;
}

export const Streamer: FunctionalComponent<StreamerProps> = ({
    sessionOpen,
    onStart,
    onStop,
    onFlush,
    isStreaming,
}) => {
    const [text, setText] = useState('Встретил он по дороге Зайчика, Волка и Медведя, спел им свою песенку и убежал от них. Катился дальше Колобок и повстречалась ему Лисичка. Она и говорит: Колобок, Колобок, какая у тебя красивая песенка.');
    const [mode, setMode] = useState<ChunkMode>('words');
    const [minSize, setMinSize] = useState(3);
    const [maxSize, setMaxSize] = useState(8);
    const [interval, setInterval] = useState(100);

    return (
        <div class="card">
            <h3>2. LLM Simulation</h3>
            <div class="row">
                <div>
                    <label class="small">Mode</label>
                    <select value={mode} onChange={(e) => setMode((e.target as HTMLSelectElement).value as ChunkMode)}>
                        <option value="tokens">Tokens</option>
                        <option value="words">Words</option>
                        <option value="chars">Chars</option>
                    </select>
                </div>
                <div>
                    <label class="small">Min</label>
                    <input type="number" value={minSize} onInput={(e) => setMinSize(parseInt((e.target as HTMLInputElement).value))} />
                </div>
                <div>
                    <label class="small">Max</label>
                    <input type="number" value={maxSize} onInput={(e) => setMaxSize(parseInt((e.target as HTMLInputElement).value))} />
                </div>
                <div>
                    <label class="small">Interval (ms)</label>
                    <input type="number" value={interval} onInput={(e) => setInterval(parseInt((e.target as HTMLInputElement).value))} />
                </div>
            </div>
            <textarea
                style="margin-top: 10px; height: 150px"
                value={text}
                onInput={(e) => setText((e.target as HTMLTextAreaElement).value)}
            />
            <div class="row" style="margin-top: 10px;">
                <button
                    class="primary"
                    onClick={() => onStart(text, { mode, minSize, maxSize, interval })}
                    disabled={!sessionOpen || isStreaming}
                >
                    Start Streaming
                </button>
                <button onClick={onStop} disabled={!isStreaming}>Stop</button>
                <button onClick={onFlush} disabled={!sessionOpen}>Force Flush</button>
            </div>
        </div>
    );
};
