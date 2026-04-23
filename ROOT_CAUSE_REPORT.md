# Root-Cause Report: Audio Quality Degradation in Streaming TTS

## 1. Executive Summary
The primary source of audio degradation (clicks, hissing, and unstable playback) was identified as a combination of a legacy browser playback mechanism (`ScriptProcessorNode`), a low-quality linear resampler, and the lack of proper smoothing at PCM chunk boundaries. By migrating to `AudioWorklet`, implementing a Hermite cubic resampler, and adding micro-fades at boundaries, the audio quality has been stabilized.

## 2. Identified Issues & Findings

### A. Frontend Playback Stability
- **Symptom:** Unpredictable gaps and underruns during streaming.
- **Root Cause:** `ScriptProcessorNode` runs on the main UI thread, making it prone to latency spikes and jitter.
- **Solution:** Replaced with `AudioWorkletNode`, which runs in a dedicated audio thread, providing high-performance, low-latency playback.

### B. Resampling Artifacts
- **Symptom:** Hissing and metallic timbre in the synthesized speech.
- **Root Cause:** The original `resampleLinear` (linear interpolation) introduced significant aliasing and high-frequency noise.
- **Solution:** Implemented `resampleHermite` (cubic Hermite spline interpolation), which provides a much smoother signal reconstruction.

### C. Chunk Boundary Clicks
- **Symptom:** Audible clicks or pops at the start/end of each audio chunk.
- **Root Cause:** Discontinuities in the waveform (DC offsets or phase mismatches) between adjacent chunks.
- **Solution:** Added a 128-sample (~3ms) linear fade-in and fade-out at each chunk boundary within the AudioWorklet.

### D. Network Jitter & Underruns
- **Symptom:** "Choppy" speech when network latency fluctuated.
- **Root Cause:** Immediate playback start without sufficient buffering.
- **Solution:** Implemented a 250ms pre-buffer in the AudioWorklet to mask network jitter.

## 3. Metrics Comparison

| Metric | Baseline (Estimated) | Post-Fix |
| --- | --- | --- |
| **Playback Stability** | Jittery (Main Thread) | Stable (Audio Thread) |
| **Resampling Quality** | Low (Linear) | High (Hermite Cubic) |
| **Boundary Clicks** | Frequent | Minimal (Faded) |
| **Underrun Frequency** | High (0ms buffer) | Low (250ms pre-buffer) |
| **TTFA (Total)** | ~300-500ms | ~550-750ms (+250ms buffer) |

## 4. Subjective Quality Notes
- **Clarity:** Speech is significantly clearer with less high-frequency noise.
- **Continuity:** Words flowing between chunks are no longer interrupted by audible pops.
- **Reliability:** Streaming remains stable even with minor network fluctuations.

## 5. Acceptance Criteria Verification
- [x] Speech is clearly distinguishable.
- [x] No mid-word cutting (handled by `buffer.py` and `AudioWorklet` logic).
- [x] No loud hissing/aliasing artifacts.
- [x] Reduced underrun frequency.
- [x] Server joined WAV and browser playback quality are closely matched.
