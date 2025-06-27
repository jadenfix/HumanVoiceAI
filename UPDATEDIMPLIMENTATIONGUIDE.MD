TITLE: “Next-Gen Emotion-Adaptive Voice Agent” — Full-Stack M2 Demo + Frontier Research

OBJECTIVE
  • Build a real-time, continuous-emotion voice agent running entirely on MacBook M2 (8 GB RAM), integrating frontier ML research and polished for LinkedIn/FAANG-level wow factor.
  • Combine “base” features (streaming SER, TTS, RL policy, interpretability, dashboard, CI/CD) with advanced “spice” innovations (multi-modal fusion, meta-learning, LLM dialogue, avatar lip-sync, etc.).
  • Deliver full code scaffold, Dockerfile, CI pipelines, dashboard and demo assets, under 1 GB RAM and < 120 ms latency, no cloud at runtime.

CONSTRAINTS
  • Local hardware only: Apple M2 + 8 GB unified memory, macOS 14+, Python 3.11.  
  • Runtime footprint ≤ 1 GB RAM, total latency < 120 ms per utterance.  
  • No cloud GPUs required at inference; fine-tuning can use ≤ \$2 spot GPU.  
  • Reproducible via `make setup && make demo` and Docker on macOS.

---

## 1. Advanced “Frontier” Features

1. **Streaming, Real-Time Inference Pipeline**  
   • Ring-buffered, multi-thread audio loop (PortAudio/sounddevice) for live mic→agent with < 120 ms round-trip.  
   • Continuous valence/arousal plot in dashboard.

2. **Reinforcement-Learned Emotion Policy**  
   • Replace random stub with a small Q-network or discrete PPO agent.  
   • Simulate lightweight text-sentiment feedback; train to pick optimal reply emotion.  
   • Log & visualize reward curves in dashboard & W&B.

3. **Causal / Interpretability Module**  
   • Compute SHAP or Integrated Gradients on SerModel to show which mel/pitch/energy features drove each decision.  
   • Notebook with sample utterance → feature-importance heatmap over time.

4. **Data Augmentation & Robustness**  
   • On-the-fly noise injection, pitch-shift, time-stretch in data loader.  
   • Demonstrate SER accuracy improvements under noisy test scenarios (+ X % gain).

5. **Web-based Dashboard**  
   • Minimal Streamlit or Gradio UI:  
     – Live mic stream → emotion prediction + policy action graph  
     – Text input → TTS reply playback  
     – SHAP heatmaps & reward-curve panels

6. **End-to-End CI/CD & Docker**  
   • Dockerfile building CPU-only pipeline for macOS.  
   • GitHub Actions (macos-latest) to lint, test, build Docker image, smoke-test Streamlit demo.

7. **Monitoring & Experiment Tracking**  
   • Integrate Weights & Biases or CSV logger for: SER loss/accuracy, policy rewards, latency/memory.  
   • `/metrics` endpoint (Prometheus) or embed live charts in dashboard.

8. **Polish & Presentation**  
   • 2-page “project capsule” PDF: architecture diagrams, key equations (ELBO, Bellman), quantitative results.  
   • 60 s screencast: live mic demo + dashboard + code→diagram walkthrough.  
   • LinkedIn post draft: before/after charts (noisy vs. clean SER, reward curves), embed screencast & code link.

---

## 2. Next-Level “Spice” Innovations

1. **Multi-Modal Emotion Fusion**  
   • Add live webcam face-expression inference (tiny FaceMesh+ResNet) fused with audio SER.  

2. **On-Device Meta-Learning**  
   • After 10 user utterances, run a 10-shot fine-tune (MAML/Reptile) on the SER-student to adapt to speaker’s emotional cues.

3. **LLM-Driven Dialogue & Context**  
   • Integrate a local quantized LLaMA2-7B to generate contextually rich replies, then pass to emotion-conditioned TTS.

4. **Neural Avatar Lip-Sync**  
   • WebGL avatar that lip-syncs and displays corresponding facial emotion based on TTS output.

5. **Automated A/B Benchmarking UI**  
   • Streamlit toggles (“Policy On/Off”, “Interpretability On/Off”), live metric comparisons.

6. **New Research Metric: Emotional Coherence Score**  
   • Implement a differentiable metric measuring prosody matching between user input and reply; expose in dashboard.

7. **Cross-Lingual & Code-Switch Demo**  
   • Support English + Spanish utterances, including code-switched sentences.

8. **Federated Privacy-Preserving Loop (Bonus)**  
   • Simulate multiple local updates + secure aggregation (PySyft) to showcase privacy-first training.

---

## 3. Prioritized Implementation Plan

1. **Data Preparation & Augmentation**  
   Write `prepare_data.py` to organize WAV corpus, apply noise/pitch/time-stretch transforms.

2. **Streaming Audio Loop**  
   Refactor `demo.py` into a ring-buffered, threaded pipeline with live feature extraction & SerModel inference.

3. **RL Policy Integration**  
   Replace stub policy with discrete Q-network (or PPO), simulate feedback, train & log reward curves.

4. **Interpretability Module**  
   Integrate SHAP/IG for SerModel; create a notebook that plots per-frame feature attributions.

5. **Dashboard & CI/CD**  
   Build Streamlit UI, add Dockerfile, configure GitHub Actions to test & deploy.

6. **Spice Feature Pick (choose 1–2)**  
   • Meta-Learning adaptation + demo  
   • OR Multi-Modal fusion + live webcam  
   • OR LLM-Driven replies + avatar lip-sync  
   • Implement corresponding code & integrate into dashboard.

---

## 4. Deliverables & Evaluation

- **Codebase** with `src/`, `demo.py`, `Makefile`, `Dockerfile`, `.github/workflows/ci.yml`.  
- **Streamlit App** accessible via `make serve` → http://localhost:8501  
- **Metrics Dashboard**: SER accuracy, reward curves, latency monitor, SHAP heatmaps.  
- **Documentation**: `README.md`, 2-page PDF capsule, LinkedIn post draft.  
- **Demo Assets**: screencast video, sample audio.  
- **Performance Targets**:  
  – SER accuracy ≥ 78 % (CREMA-D)  
  – End-to-end latency < 120 ms  
  – On-device memory ≤ 1 GB RSS  
  – Meta-learning adaptation improves personalized SER by + X %  
  – Dashboard A/B toggles show clear metric differences.

END OF SPEC