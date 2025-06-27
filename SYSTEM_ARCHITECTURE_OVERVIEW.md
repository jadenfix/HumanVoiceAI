# System Architecture Overview  
*Running entirely on Mac M2 (8 GB RAM), <120 ms end-to-end latency, ≤1 GB RSS*

---

## 1. Component Diagram

```mermaid
graph LR
  A[🎤 Audio In (PortAudio)] --> B[FeatureExtractor]
  B --> C[SerModel (CNN-BiLSTM)]
  C --> D{Policy πω} 
  D --> E[TtsEngine (XTTS-v2-small)]
  E --> F[Vocoder (HiFi-GAN-tiny)]
  F --> G[🔊 Audio Out (PortAudio)]
  C --> H[Logger (CSV)]
  D --> H
```

## 2. Data Flow & Latency Budgets

| Stage | Compute | RAM | Time |
|-------|---------|-----|------|
| Audio capture (512-frame) | PortAudio ring buffer | 50 MB | 4 ms |
| FeatureExtractor | 80-mel + F0 + energy (MPS) | 150 MB | 15 ms |
| SerModel | CNN-BiLSTM 3M params (MPS) | 200 MB | 20 ms |
| Policy πω | 2-layer MLP (CPU) | 30 MB | 1 ms |
| TtsEngine | XTTS-small-8bit (MPS) | 480 MB | 50 ms |
| Vocoder | HiFi-GAN-tiny (CoreML int8) | 50 MB | 30 ms |
| **Total (approx)** | | **~960 MB** | **~120 ms** |

> Note: We stay under ~1 GB resident memory to avoid swapping on 8 GB machines.

## 3. Subsystems & Interfaces

### 3.1 FeatureExtractor
- **Input**: 16 kHz mono WAV
- **Output**: Tensor[T, 82] (80-mel + F0 + energy)
- **Libs**: torchaudio.transforms, custom CUDA kernels for F0

### 3.2 SerModel
- **Type**: CNN-BiLSTM (~3 M parameters)
- **Input**: Tensor[T,82]
- **Output**: Tensor[T, E] (E emotion logits)
- **Backend**: PyTorch-MPS

### 3.3 Policy Module
- **Type**: 2-layer MLP
- **Input**: aggregated emotion stats + fixed context
- **Output**: discrete token ∈ {neutral, calm, happy, sad, angry, fearful}
- **Compute**: CPU-only (tiny footprint)

### 3.4 TtsEngine
- **Model**: tts_models/en/xtts_v2_8bit
- **Input**: text + emotion token
- **Output**: mel-spectrogram frames

### 3.5 Vocoder
- **Model**: HiFi-GAN-tiny (quantized for Core ML)
- **Input**: mel frames
- **Output**: PCM waveform

## 4. Logging & Debug
- Logger writes CSV lines:
  ```
  timestamp,text,in_emotion,out_emotion,latency_ms
  ```
- Add a simple local web UI with Streamlit (optional) to plot real-time latency & emotions.

## 5. CI/CD & Repro
- **Makefile targets**:
  - `make lint` (black/isort/flake8)
  - `make test` (pytest)
  - `make demo TEXT="Hi"` (runs full pipeline on sample)
- GitHub Actions via macos-latest runner to ensure Mac compatibility.
