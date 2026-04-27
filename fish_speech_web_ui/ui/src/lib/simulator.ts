export type ChunkMode = 'chars' | 'words' | 'sentences';

export interface SimulatorOptions {
  mode: ChunkMode;
  minSize: number;
  maxSize: number;
  interval: number;
  onChunk: (chunk: string) => Promise<boolean> | boolean;
  onFinish: () => Promise<void> | void;
}

export class LLMSimulator {
  private timer: number | null = null;
  private offset = 0;
  private text = '';
  private stopped = false;
  private options: SimulatorOptions;

  constructor(options: SimulatorOptions) {
    this.options = options;
  }

  start(text: string) {
    this.stop();
    this.text = text;
    this.offset = 0;
    this.stopped = false;
    void this.tick();
  }

  stop() {
    this.stopped = true;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private async finishOnce() {
    if (this.stopped) return;
    this.stopped = true;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    await this.options.onFinish();
  }

  private async tick() {
    if (this.stopped) return;

    if (this.offset >= this.text.length) {
      await this.finishOnce();
      return;
    }

    const remaining = this.text.slice(this.offset);
    const size = Math.max(
      1,
      Math.floor(
        Math.random() * (this.options.maxSize - this.options.minSize + 1) +
          this.options.minSize
      )
    );

    let chunk = '';

    if (this.options.mode === 'chars') {
      chunk = remaining.slice(0, size);
    } else if (this.options.mode === 'words') {
      const parts = remaining.match(/\S+\s*/g) || [remaining];
      chunk = parts.slice(0, size).join('');
    } else {
      const parts = remaining.match(/[^.!?…]+[.!?…]?\s*/g) || [remaining];
      chunk = parts.slice(0, size).join('');
    }

    if (!chunk) {
      chunk = remaining.slice(0, 1);
    }

    this.offset += chunk.length;

    const shouldContinue = await this.options.onChunk(chunk);
    if (!shouldContinue || this.stopped) {
      this.stop();
      return;
    }

    if (this.offset >= this.text.length) {
      await this.finishOnce();
      return;
    }

    this.timer = window.setTimeout(() => {
      void this.tick();
    }, this.options.interval);
  }
}