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
  private scheduledUntil = 0;
  private started = false;
  private pendingBuffers: Float32Array[] = [];
  private pendingSamplesCount = 0;

  constructor(
    private onStatusChange: (status: string) => void,
    private onEvent: (event: StreamEvent) => void
  ) {}

  async init() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)({
        latencyHint: 'interactive'
      });
      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.value = 1;
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

    const response = await fetch(url, { signal });
    if (!response.ok || !response.body) {
      throw new Error(`Stream connect failed: ${response.status} ${response.statusText}`);
    }

    this.onStatusChange('connected');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let textBuffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        textBuffer += decoder.decode(value, { stream: true });
        const lines = textBuffer.split('\n');
        textBuffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          const event = JSON.parse(trimmed) as StreamEvent;
          this.onEvent(event);
          await this.handleEvent(event);
        }
      }
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        this.onStatusChange('aborted');
        return;
      }

      this.onStatusChange('error');
      throw error;
    }
  }

  private async handleEvent(event: StreamEvent) {
    if (event.type === 'session_start') {
      this.onStatusChange('stream ready');
    }

    if (event.type === 'meta') {
      this.meta = {
        sample_rate: event.sample_rate,
        channels: event.channels,
        sample_width: event.sample_width
      };
      this.onStatusChange(`${this.meta.sample_rate}Hz / ${this.meta.channels}ch`);
    }

    if (event.type === 'pcm') {
      await this.handlePcm(event.data);
    }

    if (event.type === 'session_done') {
      this.onStatusChange('finished');
    }

    if (event.type === 'session_aborted') {
      this.onStatusChange('aborted');
    }

    if (event.type === 'error') {
      this.onStatusChange('error');
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
    this.pendingSamplesCount += floatData.length / Math.max(1, this.meta.channels);

    this.schedulePlayback();
  }

  private pcm16ToFloat32(bytes: Uint8Array): Float32Array {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = Math.floor(bytes.byteLength / 2);
    const output = new Float32Array(samples);

    for (let i = 0; i < samples; i++) {
      output[i] = Math.max(-1, Math.min(1, view.getInt16(i * 2, true) / 32768));
    }

    return output;
  }

  private schedulePlayback() {
    if (!this.ctx || !this.meta || !this.gainNode || this.pendingBuffers.length === 0) return;

    const channels = Math.max(1, this.meta.channels);
    const now = this.ctx.currentTime;

    if (!this.started) {
      const bufferedMs = (this.pendingSamplesCount / this.meta.sample_rate) * 1000;
      if (bufferedMs < 90) return;

      this.started = true;
      this.scheduledUntil = now + 0.08;
      this.onStatusChange('playing');
    }

    while (this.pendingBuffers.length > 0) {
      const interleaved = this.pendingBuffers.shift();
      if (!interleaved) break;

      const frames = Math.floor(interleaved.length / channels);
      this.pendingSamplesCount -= frames;

      const audioBuffer = this.ctx.createBuffer(channels, frames, this.meta.sample_rate);

      for (let channel = 0; channel < channels; channel++) {
        const channelData = audioBuffer.getChannelData(channel);
        for (let frame = 0; frame < frames; frame++) {
          channelData[frame] = interleaved[frame * channels + channel] || 0;
        }
      }

      const source = this.ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);

      const startTime = Math.max(now + 0.02, this.scheduledUntil);
      source.start(startTime);
      this.scheduledUntil = startTime + audioBuffer.duration;
    }
  }
}