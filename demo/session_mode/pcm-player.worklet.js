class PCMPlayerWorklet extends AudioWorkletProcessor {
  constructor() {
    super();

    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.queuedSamples = 0;

    this.expectMore = false;
    this.prebufferSamples = 0;
    this.prebuffering = true;
    this.started = false;
    this.rebuffering = false;
    this.lastStatsAt = -1;

    this.port.onmessage = (event) => {
      const data = event.data || {};

      if (data.type === "push" && data.payload) {
        this.queue.push(data.payload);
        this.queuedSamples += data.payload.length;
      } else if (data.type === "expect_more") {
        this.expectMore = !!data.value;
      } else if (data.type === "configure") {
        this.prebufferSamples = Math.max(0, Number(data.prebufferSamples || 0));
      } else if (data.type === "reset") {
        this.queue = [];
        this.current = null;
        this.offset = 0;
        this.queuedSamples = 0;
        this.expectMore = false;
        this.prebuffering = true;
        this.started = false;
        this.rebuffering = false;
        this.lastStatsAt = -1;
      }
    };
  }

  postStats(force = false) {
    if (!force && this.lastStatsAt >= 0 && currentTime - this.lastStatsAt < 0.1) {
      return;
    }

    this.lastStatsAt = currentTime;
    this.port.postMessage({
      type: "stats",
      queuedSamples: this.queuedSamples,
      prebuffering: this.prebuffering,
      started: this.started,
      rebuffering: this.rebuffering,
    });
  }

  maybeStartPlayback() {
    const ready =
      this.queuedSamples >= this.prebufferSamples ||
      (!this.expectMore && this.queuedSamples > 0);

    if (!this.prebuffering || !ready) {
      return false;
    }

    this.prebuffering = false;

    if (this.rebuffering) {
      this.rebuffering = false;
      this.port.postMessage({ type: "event", kind: "rebuffer_finished" });
    } else if (!this.started) {
      this.started = true;
      this.port.postMessage({ type: "event", kind: "audio_play_started" });
    }

    return true;
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const channel = output[0];
    channel.fill(0);

    this.maybeStartPlayback();

    if (this.prebuffering) {
      this.postStats();
      return true;
    }

    let written = 0;

    while (written < channel.length) {
      if (!this.current) {
        if (this.queue.length === 0) {
          break;
        }
        this.current = this.queue.shift();
        this.offset = 0;
      }

      const remainingOutput = channel.length - written;
      const remainingInput = this.current.length - this.offset;
      const count = Math.min(remainingOutput, remainingInput);

      channel.set(this.current.subarray(this.offset, this.offset + count), written);

      this.offset += count;
      written += count;
      this.queuedSamples -= count;

      if (this.offset >= this.current.length) {
        this.current = null;
      }
    }

    if (written < channel.length) {
      if (this.expectMore) {
        this.prebuffering = true;
        this.rebuffering = true;
        this.port.postMessage({
          type: "event",
          kind: "underrun",
          missingSamples: channel.length - written,
        });
        this.port.postMessage({ type: "event", kind: "rebuffer_started" });
      } else if (this.started) {
        this.started = false;
        this.port.postMessage({ type: "event", kind: "audio_play_finished" });
      }
    }

    if (!this.expectMore && this.queuedSamples <= 0 && !this.current && this.started) {
      this.started = false;
      this.port.postMessage({ type: "event", kind: "audio_play_finished" });
    }

    this.postStats();
    return true;
  }
}

registerProcessor("pcm-player", PCMPlayerWorklet);