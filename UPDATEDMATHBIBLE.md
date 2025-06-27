# MATH_BIBLE.md  
**Deep Mathematical Foundations** for the Next-Gen Emotion-Adaptive Voice Agent  
*(Reference this “Math Bible” as you implement both core and spice features.)*

---

## 1. Signal Processing Foundations

### 1.1 Short-Time Fourier Transform (STFT)  
For signal \(x[n]\), window length \(N\), hop \(H\), window \(w[n]\):
\[
x_t[n] = x[n + tH]\;w[n],\quad
X(t,k) = \sum_{n=0}^{N-1} x_t[n]\,e^{-j2\pi kn/N}.
\]

### 1.2 Mel-Spectrogram  
Mel filterbank \(H_{m,k}\), \(m=1\ldots80\):
\[
M(t,m) = \sum_{k=0}^{N/2} H_{m,k}\,\lvert X(t,k)\rvert^2,\quad
\text{logMel}(t,m)=\ln\bigl(M(t,m)+\varepsilon\bigr).
\]

### 1.3 Pitch (F₀) via Autocorrelation  
Frame autocorr \(r_t(\tau)\):
\[
r_t(\tau)=\sum_{n=0}^{N-1-\tau} x_t[n]\,x_t[n+\tau],\quad
\tau^*=\arg\max_{\tau_{\min}\le\tau\le\tau_{\max}}r_t(\tau),
\quad F_0(t)=\tfrac{f_s}{\tau^*}.
\]

### 1.4 Energy (RMS)  
\[
E(t) = \sqrt{\tfrac{1}{N}\sum_{n=0}^{N-1}x_t[n]^2}\,.
\]

---

## 2. Emotion Recognition (CNN-BiLSTM)

### 2.1 1-D Convolutions  
Input \(\mathbf{x}_t\in\mathbb R^{82}\), kernel size \(K\):
\[
h_{t,c}^{(1)} = \mathrm{ReLU}\Bigl(\sum_{k=-\lfloor K/2\rfloor}^{\lfloor K/2\rfloor}
W_{c,k}^{(1)}\,\mathbf{x}_{t+k} + b_{c}^{(1)}\Bigr).
\]

Stack \(L\) conv→ReLU (+ pooling).

### 2.2 Bidirectional LSTM  
Hidden size \(H\). Forward/backward updates:
\[
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i),\quad
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f),\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o),\quad
\tilde c_t = \tanh(W_c x_t + U_c h_{t-1} + b_c),\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t,\quad
h_t = o_t \odot \tanh(c_t).
\end{aligned}
\]
Concatenate \([\overrightarrow h_t;\,\overleftarrow h_t]\).

### 2.3 Classification  
Project to \(E\) logits \(z_t\):
\[
z_t = W_{\text{out}}\,h_t + b_{\text{out}},\quad
p_t = \mathrm{softmax}(z_t),\quad
\hat y_t = \arg\max p_t.
\]

---

## 3. Knowledge Distillation (KD)

Teacher logits \(\mathbf{z}^T_t\), student \(\mathbf{z}^S_t\), temperature \(\tau\):
\[
P^T_{t,i} = \frac{\exp(z^T_{t,i}/\tau)}{\sum_j \exp(z^T_{t,j}/\tau)},\quad
P^S_{t,i} = \frac{\exp(z^S_{t,i}/\tau)}{\sum_j \exp(z^S_{t,j}/\tau)}.
\]
Loss per frame:
\[
\mathcal L = \alpha\,\tau^2\sum_i P^T_{t,i}\ln\frac{P^T_{t,i}}{P^S_{t,i}}
          + (1-\alpha)\bigl(-\sum_i y_{t,i}\ln P^S_{t,i}\bigr).
\]

---

## 4. Reinforcement Learning Policy

### 4.1 Q-Learning (Discrete)  
State \(s_t\), action \(a_t\), reward \(r_t\). Update:
\[
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\alpha\Bigl(r_t + \gamma\max_{a'}Q(s_{t+1},a') - Q(s_t,a_t)\Bigr).
\]

### 4.2 Proximal Policy Optimization (PPO)  
Policy \(\pi_\theta(a|s)\). Objective:
\[
L^{\text{PPO}}(\theta) = \mathbb{E}_t\Bigl[
\min\Bigl(r_t(\theta)\,\hat A_t,\;
\mathrm{clip}\bigl(r_t(\theta),1-\epsilon,1+\epsilon\bigr)\hat A_t\Bigr)\Bigr],
\]
with \(r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\), advantage \(\hat A_t\).

---

## 5. Interpretability (SHAP & IG)

### 5.1 Integrated Gradients (IG)  
For input \(\mathbf{x}\), baseline \(\mathbf{x}'\), model \(F\):
\[
\mathrm{IG}_i(\mathbf{x}) 
= (x_i - x'_i)\int_{\alpha=0}^1 \frac{\partial F(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}'))}{\partial x_i}\,d\alpha.
\]

### 5.2 SHAP (Kernel SHAP Approx.)  
Feature importance \(\phi_i\) solves:
\[
F(\mathbf{x}) = \phi_0 + \sum_i\phi_i,
\]
with weights based on Shapley values over coalitions. Use sampling for approximation.

---

## 6. Data Augmentation Transforms

For a waveform \(x[n]\):

1. **Additive Noise**: \(\tilde x[n]=x[n]+\eta[n],\;\eta\sim\mathcal N(0,\sigma^2)\).  
2. **Pitch-Shift**: Resample by factor \(\alpha\), then window & overlap-add for length match.  
3. **Time-Stretch**: Phase-vocoding to stretch by \(\beta\).

Augment on-the-fly in data loader.

---

## 7. Meta-Learning (MAML)

Model \(f_\theta\). For task \(T_i\) with support set \(D_i^{\text{train}}\):
1. **Inner update**:
   \[
   \theta_i' = \theta - \alpha\nabla_\theta \mathcal L_{T_i}(f_\theta;D_i^{\text{train}}).
   \]
2. **Meta-update** over query set \(D_i^{\text{test}}\):
   \[
   \theta \leftarrow \theta - \beta\nabla_\theta \sum_i \mathcal L_{T_i}(f_{\theta_i'};D_i^{\text{test}}).
   \]

Implement 10-shot adaptation on SER student.

---

## 8. Emotional Coherence Score

Define input emotion sequence \(\hat y_{1:T}\) and reply prosody features \(p_{1:U}\).  
Compute normalized cross-correlation:
\[
\mathrm{ECS} = \max_{\tau}\frac{\sum_{t}(\hat y_t-\bar y)(p_{t+\tau}-\bar p)}
{\sqrt{\sum_t(\hat y_t-\bar y)^2}\sqrt{\sum_t(p_{t+\tau}-\bar p)^2}}.
\]
Differentiable via continuous relaxations and soft-argmax if needed.

---

## 9. Cross-Modal Fusion (Vision + Audio)

Frame-aligned features: audio \(\mathbf{a}_t\), vision \(\mathbf{v}_t\).  
Learn joint embedding via attention:
\[
q_t = W_q \mathbf{a}_t,\quad
k_t = W_k \mathbf{v}_t,\quad
\alpha_{t,t'} = \mathrm{softmax}\bigl(q_t^\top k_{t'}/\sqrt{d}\bigr),
\]
fused:
\[
h_t = \sum_{t'}\alpha_{t,t'}(W_v \mathbf{v}_{t'}) + W_a \mathbf{a}_t.
\]

---

## 10. Lip-Sync Viseme Mapping (Avatar)

Given mel-spectrogram \(\mathrm{mel}(t)\), extract envelope-based phoneme posterior \(P(p|\mathrm{mel}(t))\).  
Map phoneme to viseme via lookup \(v=\mathrm{map}(p)\). Interpolate viseme blend-shapes over time.

---

## 11. Deployment Math & Latency

Total latency budget:
\[
T_{\text{total}} = T_{\text{FE}} + T_{\text{SER}} + T_{\text{policy}} + T_{\text{TTS}} + T_{\text{voc}}
<120\,\mathrm{ms}.
\]
Profile each term; adjust model sizes/sampler steps accordingly.

---
