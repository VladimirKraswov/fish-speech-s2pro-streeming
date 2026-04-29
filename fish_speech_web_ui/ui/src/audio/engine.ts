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

  private activeSources = new Set<AudioBufferSourceNode>();

  private initialPlaybackBufferMs = 180;
  private initialStartDelaySec = 0.05;

  // Если до конца уже запланированного аудио осталось меньше этого,
  // новый source лучше не приклеивать впритык — браузер может не успеть.
  private readonly criticalLeadSec = 0.045;

  // Задержка восстановления после underrun.
  private readonly underrunRestartDelaySec = 0.18;

  private lastStatus = '';
  private lastStatusAt = 0;

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
      const AudioContextCtor =
        window.AudioContext || (window as any).webkitAudioContext;

      this.ctx = new AudioContextCtor({
        // Чуть стабильнее, чем latencyHint: 'interactive'.
        // Для TTS важнее ровный звук, чем минимальная задержка.
        latencyHint: 0.18,
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
    for (const source of this.activeSources) {
      try {
        source.stop();
      } catch {
        // Source may already be stopped or not started yet.
      }
    }

    this.activeSources.clear();

    this.meta = null;
    this.scheduledUntil = 0;
    this.started = false;

    this.pendingBuffers = [];
    this.pendingSamplesCount = 0;

    // We keep the configured initialPlaybackBufferMs and initialStartDelaySec
    // as they were set for the session.

    this.setStatus('idle', true);
  }

  configurePlayback(options: {
    clientStartBufferMs?: number;
    clientInitialStartDelayMs?: number;
  }) {
    if (typeof options.clientStartBufferMs === 'number') {
      this.initialPlaybackBufferMs = Math.max(0, options.clientStartBufferMs);
    }

    if (typeof options.clientInitialStartDelayMs === 'number') {
      this.initialStartDelaySec = Math.max(0, options.clientInitialStartDelayMs) / 1000;
    }
  }

  async connectStream(url: string, signal: AbortSignal) {
    await this.init();
    this.reset();

    const response = await fetch(url, {
      signal,
      cache: 'no-store',
    });

    if (!response.ok || !response.body) {
      throw new Error(`Stream connect failed: ${response.status} ${response.statusText}`);
    }

    this.setStatus('connected', true);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let textBuffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          textBuffer += decoder.decode();
          break;
        }

        textBuffer += decoder.decode(value, { stream: true });

        const lines = textBuffer.split('\n');
        textBuffer = lines.pop() || '';

        for (const line of lines) {
          await this.handleStreamLine(line);
        }
      }

      if (textBuffer.trim()) {
        await this.handleStreamLine(textBuffer);
      }
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        this.setStatus('aborted', true);
        return;
      }

      this.setStatus('error', true);
      throw error;
    }
  }

  private async handleStreamLine(line: string) {
    const trimmed = line.trim();
    if (!trimmed) return;

    let event: StreamEvent;

    try {
      event = JSON.parse(trimmed) as StreamEvent;
    } catch {
      this.setStatus('bad stream event');
      return;
    }

    this.onEvent(event);
    await this.handleEvent(event);
  }

  private async handleEvent(event: StreamEvent) {
    if (event.type === 'session_start') {
      this.configurePlayback({
        clientStartBufferMs: event.client_start_buffer_ms,
        clientInitialStartDelayMs: event.client_initial_start_delay_ms,
      });

      this.setStatus('stream ready', true);
      return;
    }

    if (event.type === 'meta') {
      this.meta = {
        sample_rate: event.sample_rate,
        channels: event.channels,
        sample_width: event.sample_width,
      };

      this.setStatus(`${this.meta.sample_rate}Hz / ${this.meta.channels}ch`, true);
      return;
    }

    if (event.type === 'pcm') {
      await this.handlePcm(event.data);
      return;
    }

    if (event.type === 'session_done') {
      this.setStatus('finished', true);
      return;
    }

    if (event.type === 'session_aborted') {
      this.setStatus('aborted', true);
      return;
    }

    if (event.type === 'error') {
      this.setStatus('error', true);
    }
  }

  private async handlePcm(base64Data: string) {
    if (!this.meta || !this.ctx) return;

    const bytes = this.base64ToBytes(base64Data);

    if (bytes.byteLength < 2) {
      return;
    }

    const floatData = this.pcm16ToFloat32(bytes);

    if (floatData.length === 0) {
      return;
    }

    this.pendingBuffers.push(floatData);
    this.pendingSamplesCount += floatData.length / Math.max(1, this.meta.channels);

    this.schedulePlayback();
  }

  private base64ToBytes(base64Data: string): Uint8Array {
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    return bytes;
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

  private mergePendingBuffers(): Float32Array | null {
    if (this.pendingBuffers.length === 0) {
      return null;
    }

    if (this.pendingBuffers.length === 1) {
      const only = this.pendingBuffers.shift() || null;
      this.pendingSamplesCount = 0;
      return only;
    }

    let totalLength = 0;

    for (const buffer of this.pendingBuffers) {
      totalLength += buffer.length;
    }

    const merged = new Float32Array(totalLength);
    let offset = 0;

    for (const buffer of this.pendingBuffers) {
      merged.set(buffer, offset);
      offset += buffer.length;
    }

    this.pendingBuffers = [];
    this.pendingSamplesCount = 0;

    return merged;
  }

  private schedulePlayback() {
    if (!this.ctx || !this.meta || !this.gainNode || this.pendingBuffers.length === 0) {
      return;
    }

    const channels = Math.max(1, this.meta.channels);
    const now = this.ctx.currentTime;

    if (!this.started) {
      const bufferedMs = (this.pendingSamplesCount / this.meta.sample_rate) * 1000;

      if (bufferedMs < this.initialPlaybackBufferMs) {
        this.setStatus(`buffering ${Math.round(bufferedMs)}ms`);
        return;
      }

      this.started = true;
      this.scheduledUntil = now + this.initialStartDelaySec;
      this.setStatus(`playing, buffer=${Math.round(bufferedMs)}ms`, true);
    }

    const leadSec = this.scheduledUntil - now;

    if (leadSec < this.criticalLeadSec) {
      this.scheduledUntil = now + this.underrunRestartDelaySec;
      this.setStatus('underrun recovery', true);
    }

    const interleaved = this.mergePendingBuffers();
    if (!interleaved) return;

    const frames = Math.floor(interleaved.length / channels);
    if (frames <= 0) return;

    const usableSamples = frames * channels;
    const audioBuffer = this.ctx.createBuffer(channels, frames, this.meta.sample_rate);

    for (let channel = 0; channel < channels; channel++) {
      const channelData = audioBuffer.getChannelData(channel);

      for (let frame = 0; frame < frames; frame++) {
        const sampleIndex = frame * channels + channel;

        if (sampleIndex < usableSamples) {
          channelData[frame] = interleaved[sampleIndex] || 0;
        } else {
          channelData[frame] = 0;
        }
      }
    }

    const source = this.ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.gainNode);

    source.onended = () => {
      this.activeSources.delete(source);
    };

    const startTime = Math.max(this.scheduledUntil, now + this.criticalLeadSec);

    this.activeSources.add(source);
    source.start(startTime);

    this.scheduledUntil = startTime + audioBuffer.duration;
  }

  private setStatus(status: string, force = false) {
    const now = performance.now();

    if (!force && status === this.lastStatus && now - this.lastStatusAt < 300) {
      return;
    }

    if (!force && status.startsWith('buffering') && now - this.lastStatusAt < 250) {
      return;
    }

    this.lastStatus = status;
    this.lastStatusAt = now;
    this.onStatusChange(status);
  }
}
