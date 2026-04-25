import type { FunctionalComponent } from 'preact';
import { useRef, useEffect } from 'preact/hooks';

interface EventLogProps {
    logs: string[];
}

export const EventLog: FunctionalComponent<EventLogProps> = ({ logs }) => {
    const logRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div class="card" style="margin-top: 16px;">
            <h3>Log</h3>
            <div
                ref={logRef}
                class="pane mono"
                style="height: 250px; background: #111; color: #ddd; overflow-y: auto;"
            >
                {logs.map((log, i) => (
                    <div key={i}>{log}</div>
                ))}
            </div>
        </div>
    );
};
