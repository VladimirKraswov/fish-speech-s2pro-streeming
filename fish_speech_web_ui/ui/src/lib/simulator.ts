// /ui/src/lib/simulator.ts

export type ChunkMode = 'chars' | 'words' | 'tokens';

export interface SimulatorOptions {
  mode: ChunkMode;
  minSize?: number;
  maxSize?: number;
  interval: number;           // базовая задержка между чанками (мс)
  speedMultiplier?: number;   // 1.0 = нормально, 0.6 = быстрее, 1.5 = медленнее
  onChunk: (chunk: string) => Promise<boolean> | boolean;
  onFinish: () => Promise<void> | void;
}

export class LLMSimulator {
  private timer: number | null = null;
  private text = '';
  private position = 0;
  private stopped = false;
  private options: SimulatorOptions;

  constructor(options: SimulatorOptions) {
    this.options = {
      minSize: 1,
      maxSize: 3,
      speedMultiplier: 1.0,
      ...options,
    };
  }

  start(fullText: string) {
    this.stop();
    this.text = fullText;
    this.position = 0;
    this.stopped = false;
    this.scheduleNext();
  }

  stop() {
    this.stopped = true;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private getNextChunk(): string {
    if (this.position >= this.text.length) return '';

    const remaining = this.text.slice(this.position);

    if (this.options.mode === 'chars') {
      const size = Math.floor(Math.random() * 3) + 1; // 1–3 символа
      const chunk = remaining.slice(0, size);
      this.position += chunk.length;
      return chunk;
    }

    if (this.options.mode === 'words') {
      // Берём от 1 до 4 слов за раз
      const words = remaining.match(/\S+\s*/g) || [remaining];
      const count = Math.min(
        Math.floor(Math.random() * 4) + 1,
        words.length
      );
      const chunk = words.slice(0, count).join('');
      this.position += chunk.length;
      return chunk;
    }

    // Режим "tokens" — самый реалистичный (рекомендуется)
    if (this.options.mode === 'tokens') {
      // Имитируем генерацию токенов: иногда 1 символ, иногда слог/слово, иногда пунктуацию
      let chunk = '';

      // С вероятностью иногда выдаём целое слово
      if (Math.random() < 0.35 && remaining.match(/^\S+/)) {
        const wordMatch = remaining.match(/^\S+\s*/);
        if (wordMatch) {
          chunk = wordMatch[0];
        }
      } 
      // Иначе — небольшие кусочки (1-4 символа)
      else {
        const len = Math.floor(Math.random() * 4) + 1;
        chunk = remaining.slice(0, len);
      }

      this.position += chunk.length;
      return chunk;
    }

    return '';
  }

  private getDelay(): number {
    const base = this.options.interval;
    const variation = base * 0.4; // ±40% случайности
    let delay = base + (Math.random() * variation * 2 - variation);

    // Иногда делаем паузу подлиннее (как будто модель "думает")
    if (Math.random() < 0.08) {
      delay += base * 1.8;
    }

    return Math.max(16, Math.floor(delay * (this.options.speedMultiplier || 1)));
  }

  private async scheduleNext() {
    if (this.stopped || this.position >= this.text.length) {
      if (!this.stopped) {
        await this.finishOnce();
      }
      return;
    }

    const chunk = this.getNextChunk();

    if (!chunk) {
      await this.finishOnce();
      return;
    }

    const shouldContinue = await this.options.onChunk(chunk);

    if (!shouldContinue || this.stopped) {
      this.stop();
      return;
    }

    this.timer = window.setTimeout(() => {
      void this.scheduleNext();
    }, this.getDelay());
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
}