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
    private options: SimulatorOptions;

    constructor(options: SimulatorOptions) {
        this.options = options;
    }

    start(text: string) {
        this.stop();
        this.text = text;
        this.offset = 0;
        this.tick();
    }

    stop() {
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
    }

    private async tick() {
        if (this.offset >= this.text.length) {
            await this.options.onFinish();
            return;
        }

        const remaining = this.text.slice(this.offset);
        let chunk = '';
        let newOffset = this.offset;

        const size = Math.floor(
            Math.random() * (this.options.maxSize - this.options.minSize + 1) + this.options.minSize
        );

        if (this.options.mode === 'chars') {
            chunk = remaining.slice(0, size);
            newOffset += chunk.length;
        } else if (this.options.mode === 'words') {
            const words = remaining.split(/(\s+)/);
            let wordCount = 0;
            let i = 0;
            while (i < words.length && wordCount < size) {
                chunk += words[i];
                if (words[i].trim()) wordCount++;
                i++;
            }
            newOffset += chunk.length;
        } else if (this.options.mode === 'sentences') {
            const sentences = remaining.split(/([.!?]+[\s\n]*)/);
            let sentenceCount = 0;
            let i = 0;
            while (i < sentences.length && sentenceCount < size) {
                chunk += sentences[i];
                if (sentences[i].trim() && !sentences[i].match(/^[.!?\s\n]+$/)) sentenceCount++;
                i++;
            }
            newOffset += chunk.length;
        }

        if (!chunk) {
            // fallback: take one character
            chunk = remaining.slice(0, 1);
            newOffset = this.offset + 1;
        }

        this.offset = newOffset;
        const shouldContinue = await this.options.onChunk(chunk);
        if (shouldContinue && this.offset < this.text.length) {
            this.timer = window.setTimeout(() => this.tick(), this.options.interval);
        } else if (!shouldContinue) {
            this.stop();
        }
    }
}