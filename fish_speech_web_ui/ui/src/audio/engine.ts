import type { StreamEvent } from '../types';

export interface AudioMetadata {
    sample_rate: number;
    channels: number;
    sample_width: number;
}

export class AudioEngine {
    private ctx: AudioContext | null = null;
    private gainNode: GainNode | null = null;
    private meta: AudioMetadata | null = null;
    private scheduledUntil: number = 0;
    private started: boolean = false;
    private pendingBuffers: Float32Array[] = [];
    private pendingSamplesCount: number = 0;
    private onStatusChange: (status: string) => void;
    private onEvent: (event: StreamEvent) => void;

    constructor(
        onStatusChange: (status: string) => void,
        onEvent: (event: StreamEvent) => void
    ) {
        this.onStatusChange = onStatusChange;
        this.onEvent = onEvent;
    }

    async init() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)({
                latencyHint: 'interactive',
            });
            this.gainNode = this.ctx.createGain();
            this.gainNode.gain.value = 1.0;
            this.gainNode.connect(this.ctx.destination);
        }
        if (this.ctx.state === 'suspended') {
            await this.ctx.resume();
        }
    }

    reset() {
        this.meta = null;
        this.scheduledUntil = 0;
        this.started = false;
        this.pendingBuffers = [];
        this.pendingSamplesCount = 0;
        this.onStatusChange('idle');
    }

    async connectStream(url: string, signal: AbortSignal) {
        await this.init();
        this.reset();

        const resp = await fetch(url, { signal });
        if (!resp.ok || !resp.body) {
            throw new Error(`Failed to connect to stream: ${resp.statusText}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.trim()) continue;
                    const event: StreamEvent = JSON.parse(line);
                    this.onEvent(event);
                    await this.handleEvent(event);
                }
            }
        } catch (e: any) {
            if (e.name === 'AbortError') {
                this.onStatusChange('aborted');
            } else {
                console.error('Stream error:', e);
                this.onStatusChange('error');
            }
        }
    }

    private async handleEvent(event: StreamEvent) {
        switch (event.type) {
            case 'session_start':
                this.onStatusChange('stream connected');
                break;
            case 'meta':
                this.meta = {
                    sample_rate: event.sample_rate,
                    channels: event.channels,
                    sample_width: event.sample_width,
                };
                this.onStatusChange(`playing @${this.meta.sample_rate}Hz`);
                break;
            case 'pcm':
                await this.handlePcm(event.data);
                break;
            case 'session_done':
                this.onStatusChange('finished');
                break;
            case 'session_aborted':
                this.onStatusChange('aborted');
                break;
        }
    }

    private async handlePcm(base64Data: string) {
        if (!this.meta || !this.ctx) return;

        const binary = atob(base64Data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }

        const floatData = this.pcm16ToFloat32(bytes);
        this.pendingBuffers.push(floatData);
        this.pendingSamplesCount += floatData.length;

        this.schedulePlayback();
    }

    private pcm16ToFloat32(bytes: Uint8Array): Float32Array {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const samples = bytes.byteLength / 2;
        const out = new Float32Array(samples);
        for (let i = 0; i < samples; i++) {
            const s = view.getInt16(i * 2, true);
            out[i] = Math.max(-1, Math.min(1, s / 32768));
        }
        return out;
    }

    private schedulePlayback() {
        if (!this.ctx || !this.meta || this.pendingBuffers.length === 0) return;

        const now = this.ctx.currentTime;
        if (!this.started) {
            // Buffer a bit before starting
            const bufferDurationMs = (this.pendingSamplesCount / this.meta.sample_rate) * 1000;
            if (bufferDurationMs < 100) return;

            this.started = true;
            this.scheduledUntil = now + 0.1;
        }

        while (this.pendingBuffers.length > 0) {
            const chunk = this.pendingBuffers.shift()!;
            this.pendingSamplesCount -= chunk.length;

            const audioBuffer = this.ctx.createBuffer(1, chunk.length, this.meta.sample_rate);
            audioBuffer.copyToChannel(chunk as any, 0);

            const source = this.ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.gainNode!);

            const startTime = Math.max(now + 0.02, this.scheduledUntil);
            source.start(startTime);
            this.scheduledUntil = startTime + audioBuffer.duration;
        }
    }
}
