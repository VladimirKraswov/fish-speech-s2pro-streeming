import type { SessionOpenResponse, SessionAppendResponse } from '../types';

export class FishProxyClient {
    private baseUrl: string;

    constructor(baseUrl: string = '') {
        // If empty, it will be relative to the page origin
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    async getHealth() {
        const resp = await fetch(`${this.baseUrl}/health`);
        if (!resp.ok) throw new Error(`Health check failed: ${resp.status}`);
        return await resp.json();
    }

    async openSession(configText: string): Promise<SessionOpenResponse> {
        const resp = await fetch(`${this.baseUrl}/session/open`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_text: configText }),
        });
        if (!resp.ok) {
            const error = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(error.detail || `Failed to open session: ${resp.status}`);
        }
        return await resp.json();
    }

    async appendText(sessionId: string, text: string): Promise<SessionAppendResponse> {
        const resp = await fetch(`${this.baseUrl}/session/${sessionId}/append`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        if (!resp.ok) {
            const error = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(error.detail || `Failed to append text: ${resp.status}`);
        }
        return await resp.json();
    }

    async flushSession(sessionId: string, reason: string = 'manual_flush'): Promise<SessionAppendResponse> {
        const resp = await fetch(`${this.baseUrl}/session/${sessionId}/flush`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reason }),
        });
        if (!resp.ok) {
            const error = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(error.detail || `Failed to flush session: ${resp.status}`);
        }
        return await resp.json();
    }

    async finishSession(sessionId: string, reason: string = 'input_finished'): Promise<SessionAppendResponse> {
        const resp = await fetch(`${this.baseUrl}/session/${sessionId}/finish`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reason }),
        });
        if (!resp.ok) {
            const error = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(error.detail || `Failed to finish session: ${resp.status}`);
        }
        return await resp.json();
    }

    async closeSession(sessionId: string): Promise<{ ok: boolean }> {
        const resp = await fetch(`${this.baseUrl}/session/close`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        if (!resp.ok) {
            const error = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(error.detail || `Failed to close session: ${resp.status}`);
        }
        return await resp.json();
    }

    getStreamUrl(sessionId: string): string {
        return `${this.baseUrl}/session/${sessionId}/pcm-stream`;
    }
}
