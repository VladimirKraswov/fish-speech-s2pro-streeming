import type { FunctionalComponent } from 'preact';

interface SessionManagerProps {
    sessionId: string | null;
    status: string;
    configText: string;
    onConfigChange: (text: string) => void;
    onOpen: () => void;
    onFinish: () => void;
    onClose: () => void;
}

export const SessionManager: FunctionalComponent<SessionManagerProps> = ({
    sessionId,
    status,
    configText,
    onConfigChange,
    onOpen,
    onFinish,
    onClose,
}) => {
    return (
        <div class="card">
            <h3>1. Session Control</h3>
            <textarea
                class="mono"
                value={configText}
                onInput={(e) => onConfigChange((e.target as HTMLTextAreaElement).value)}
                style="height: 150px"
            />
            <div class="row" style="margin-top: 10px;">
                <button
                    class="primary"
                    onClick={onOpen}
                    disabled={status === 'opening' || status === 'open'}
                >
                    Open Session
                </button>
                <button
                    onClick={onFinish}
                    disabled={status !== 'open'}
                >
                    Finish Input
                </button>
                <button
                    onClick={onClose}
                    disabled={!sessionId}
                >
                    Close Session
                </button>
            </div>
            <div style="margin-top: 10px;">
                <strong>Status:</strong> {status}
            </div>
            {sessionId && (
                <div style="font-size: 12px; color: #666; word-break: break-all;">
                    <strong>ID:</strong> {sessionId}
                </div>
            )}
        </div>
    );
};
