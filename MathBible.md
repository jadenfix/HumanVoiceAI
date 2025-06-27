# Deep Mathematical Dive — Emotion-Adaptive Voice Agent (Mac M2, 8 GB)

> This document unpacks the core mathematical foundations you’ll implement.  
> Reference alongside your code for theoretical clarity and rigorous design.

---

## 1. Preliminaries & Notation

- **Audio signal**: \(x[n]\), sampled at \(f_s=16\,\mathrm{kHz}\).  
- **Frame shift**: \(H\) samples (e.g.\ \(H=160\) for 10 ms).  
- **Frame length**: \(N\) samples (e.g.\ \(N=400\) for 25 ms).  
- **Time index**: \(t\in\{0,1,\dots,T-1\}\) for \(T\) frames.  
- **Feature dimension**: \(D=82\) (80-band mel + F₀ + energy).  
- **Emotion classes**: \(E\) discrete tokens (e.g.\ \(E=6\)).

---

## 2. Feature Extraction

### 2.1 Short-Time Fourier Transform (STFT)

For each frame \(t\), windowed signal
\[
x_t[n] = x[n + tH]\;w[n],\quad n=0,\dots,N-1,
\]
with a Hann window \(w[n]\). Its DFT is
\[
X(t,k) \;=\; \sum_{n=0}^{N-1} x_t[n]\;e^{-j2\pi kn/N},
\quad k=0,\dots,N-1.
\]
We keep magnitudes \(\bigl|X(t,k)\bigr|\) for mel filtering.

### 2.2 Mel-Spectrogram

Define mel filterbank weights \(H_{m,k}\) for \(m=1,\dots,80\). Then
\[
M(t,m) \;=\;\sum_{k=0}^{N/2} H_{m,k}\,\bigl|X(t,k)\bigr|^2.
\]
Log-mel:
\[
\mathrm{logMel}(t,m) = \ln\bigl(M(t,m) + \varepsilon\bigr).
\]

### 2.3 Fundamental Frequency (F₀) via Autocorrelation

Compute autocorrelation \(r_t(\tau)\) of windowed frame:
\[
r_t(\tau) = \sum_{n=0}^{N-1-\tau} x_t[n]\;x_t[n+\tau],\quad \tau\in[0,\tau_{\max}].
\]
Choose lag
\(\tau^* = \arg\max_{\tau}r_t(\tau)\) within plausible pitch range (e.g.\ 50–500 Hz):
\[
F_0(t) = \frac{f_s}{\tau^*}.
\]

### 2.4 Energy

Root-mean-square energy per frame:
\[
E(t) = \sqrt{\tfrac{1}{N}\sum_{n=0}^{N-1}x_t[n]^2}\,.  
\]

---

## 3. Emotion Recognition — CNN-BiLSTM

### 3.1 Convolutional Layers

Input mel-features \(\mathbf{x}_t\in\mathbb{R}^{80}\) (plus F₀, E → \(\mathbb{R}^{82}\)). A 1-D conv layer with kernel size \(K\) computes for channel \(c\):
\[
h_{t,c}^{(1)} = \mathrm{ReLU}\Bigl(\sum_{k=-\lfloor K/2\rfloor}^{\lfloor K/2\rfloor}
  W^{(1)}_{c,k}\;\mathbf{x}_{t+k} + b^{(1)}_c\Bigr).
\]

Stack \(L\) such conv-ReLU blocks (optionally with pooling).

### 3.2 Bidirectional LSTM

Hidden size \(H\). For each time \(t\):

**Forward LSTM**:
\[
\begin{aligned}
i_t &= \sigma(W_i\,h_{t}^{\text{in}} + U_i\,h_{t-1} + b_i)\\
f_t &= \sigma(W_f\,h_{t}^{\text{in}} + U_f\,h_{t-1} + b_f)\\
o_t &= \sigma(W_o\,h_{t}^{\text{in}} + U_o\,h_{t-1} + b_o)\\
\tilde c_t &= \tanh(W_c\,h_{t}^{\text{in}} + U_c\,h_{t-1} + b_c)\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
\]
**Backward LSTM** processes the sequence in reverse.  
Concatenate \([\overrightarrow h_t;\,\overleftarrow h_t]\in\mathbb{R}^{2H}\).

### 3.3 Classification Head

Project to \(E\) emotion logits:
\[
z_t = W_{\text{out}}\,[\overrightarrow h_t;\,\overleftarrow h_t] + b_{\text{out}},
\quad z_t\in\mathbb{R}^E.
\]
Convert to probabilities:
\[
p_t = \mathrm{softmax}(z_t),\qquad p_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^E e^{z_{t,j}}}.
\]
Prediction per frame: \(\hat y_t = \arg\max_i p_{t,i}\).  
Aggregate to utterance-level via majority vote or mode.

---

## 4. Knowledge Distillation (Teacher → Student)

Let teacher logits \(\mathbf{z}^T_t\), student logits \(\mathbf{z}^S_t\). With temperature \(\tau>1\):
\[
P^T_{t,i} = \frac{\exp(z^T_{t,i}/\tau)}{\sum_j \exp(z^T_{t,j}/\tau)},\quad
P^S_{t,i} = \frac{\exp(z^S_{t,i}/\tau)}{\sum_j \exp(z^S_{t,j}/\tau)}.
\]
Distillation loss (per frame):
\[
\mathcal L_{\mathrm{KD}}(t) 
= \tau^2 \sum_{i=1}^E P^T_{t,i}\,\ln\frac{P^T_{t,i}}{P^S_{t,i}}.
\]
Include ground-truth cross-entropy:
\[
\mathcal L_{\text{CE}}(t) 
= -\sum_{i=1}^E y_{t,i}\ln P^S_{t,i},
\]
total:
\[
\mathcal L = \alpha\,\mathcal L_{\mathrm{KD}} + (1-\alpha)\,\mathcal L_{\text{CE}}.
\]

---

## 5. Policy Learning (Q-Learning on Discrete Emotions)

State \(s\) = aggregated features (e.g.\ mode of predicted frames).  
Action \(a\in\{1,\dots,E\}\) = emotion token for reply.

Define Q-value \(Q(s,a)\approx\mathbb{E}[\sum_{k=0}^\infty \gamma^k r_{t+k+1}\mid s_t=s,a_t=a]\).

**Bellman update**:
\[
Q(s_t,a_t)\;\leftarrow\;Q(s_t,a_t)
+ \alpha\Bigl(r_t + \gamma\max_{a'}Q(s_{t+1},a') - Q(s_t,a_t)\Bigr).
\]
Alternatively, train Q-network \(Q_\theta(s,a)\) via MSE loss:
\[
\mathcal L(\theta) 
= \mathbb{E}\Bigl[\bigl(y_t - Q_\theta(s_t,a_t)\bigr)^2\Bigr],\quad
y_t = r_t + \gamma\max_{a'}Q_{\theta^-}(s_{t+1},a').
\]

For on-policy, use small discrete-action PPO:
\[
\max_\theta \;\mathbb{E}_{t}\Bigl[
  \frac{\pi_\theta(a_t\!\mid s_t)}{\pi_{\theta_\text{old}}(a_t\!\mid s_t)}\,
  \hat A_t\Bigr],\quad
\hat A_t = \sum_{k=0}^\infty \gamma^k r_{t+k} - V_\phi(s_t).
\]

---

## 6. TTS & Vocoder (Black-Box)

We leverage a pretrained **XTTS-v2-small-8bit** for mel synthesis:
\[
\mathrm{melFrames} = \mathrm{XTTS}(\,\text{text},\;\text{emotionToken}\,),
\]
and a HiFi-GAN-tiny vocoder for inversion:
\[
\hat y[n] = \mathrm{Vocoder}\bigl(\mathrm{melFrames}\bigr).
\]

> **Note:** Internals of diffusion-based TTS are abstracted; focus on emotion-conditioning via discrete tokens.

---

## 7. End-to-End Loss & Metrics

- **SER accuracy**: frame-level F1, utterance-level accuracy.  
- **KD distill quality**: KL-divergence between teacher & student distributions.  
- **Policy reward**: cumulative reward per episode, average episode length.  
- **Latency**: empirical <120 ms E2E.  
- **Memory**: RSS <1 GB on M2.

---

### 📚 Key References

- Rabiner & Schafer, *Digital Processing of Speech Signals* (STFT, mel-spectrogram).  
- Graves & Schmidhuber, *Framewise Phoneme Classification with Bidirectional LSTM* (LSTM math).  
- Hinton et al., *Distilling the Knowledge in a Neural Network* (KD).  
- Watkins & Dayan, *Q-Learning* (Bellman eq.).  
- Prabhu et al., *EMOCONV-DIFF* (emotion token conditioning).

---