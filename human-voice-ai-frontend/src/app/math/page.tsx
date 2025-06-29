'use client';

import Link from 'next/link';
import { FaArrowLeft, FaWaveSquare, FaMicroscope } from 'react-icons/fa';
import { motion } from 'framer-motion';
import Equation from '../../components/Equation';

export default function MathPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-purple-950 to-fuchsia-900 text-gray-100">
      {/* Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-black/30" />
        <header className="relative z-10 container mx-auto px-6 py-6 flex items-center justify-between">
          <Link href="/" className="inline-flex items-center text-sm text-gray-300 hover:text-white">
            <FaArrowLeft className="mr-2" /> Back to App
          </Link>
          <h1 className="text-2xl font-semibold bg-gradient-to-r from-sky-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            📐 Deep&nbsp;Math&nbsp;of&nbsp;Voice&nbsp;AI
          </h1>
        </header>
      </div>

      <div className="container mx-auto px-6 py-12 space-y-12">
        {/* Signal Processing Fundamentals */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
          <h2 className="text-3xl font-bold mb-6 flex items-center">
            <FaWaveSquare className="mr-3 text-sky-400" />
            Signal Processing Pipeline
          </h2>
          <div className="space-y-6 text-gray-300 leading-relaxed">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">1️⃣ Framing &amp; Windowing</h3>
              <p>
                The discrete waveform <code>x[n]</code> is split into overlapping frames of <span className="font-semibold">25&nbsp;ms</span> with a hop of <span className="font-semibold">10&nbsp;ms</span>. A Hann window <code>w[n] = 0.5 · (1 − cos(2πn / (N−1)))</code> mitigates spectral leakage before the short-time Fourier transform (STFT).
              </p>
              <Equation value="X_m(k) = \sum_{n=0}^{N-1} x[ n + mH ]\, w[n] \, e^{-j\,2\pi k n / N}" />
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">2️⃣ Mel-Frequency Cepstral Coefficients (MFCC)</h3>
              <p>
                The magnitude spectrum is filtered by <code>M</code> triangular Mel filters. Log energies are decorrelated using the Discrete Cosine Transform (DCT) to yield cepstral coefficients <code>c_m</code>:
              </p>
              <Equation value="c_m(k) = \sum_{k=1}^K \ln|X(k)|\, \cos(m\pi k/K)\, e^{-0.5\pi m/K}" />
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">3️⃣ Spectral Descriptors</h3>
              <p>
                <strong>Centroid</strong> <code>μ</code> measures the "center of mass" of the spectrum, <strong>roll-off</strong> <code>f_r</code> encloses 85&nbsp;% of energy, and <strong>ZCR</strong> captures temporal sharpness:
              </p>
              <Equation value="mu = ( \sum_k f_k |X(k)| ) / ( \sum_k |X(k)| ), \; ZCR = (1/(N-1)) \sum_{n=1}^{N-1} 1[ x[n] x[n-1] < 0 ]" />
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">4️⃣ Embedding via wav2vec&nbsp;2.0</h3>
              <p>
                Frames are fed into a self-supervised <code>wav2vec&nbsp;2.0</code> encoder producing contextual embeddings <code>h_t ∈ ℝ<sup>768</sup></code>. A transformer with <span className="font-semibold">24</span> layers and multi-head self-attention learns latent speech representations:
              </p>
              <Equation value="SelfAttn(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V" />
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">5️⃣ Emotion Classification</h3>
              <p>
                A linear head projects <code>h_t</code> to logits <code>z ∈ ℝ<sup>C</sup></code> where <code>C = 8</code> emotions. Softmax yields posterior probabilities <code>p = softmax(z)</code>. The predicted label is <code>ŷ = argmax p</code>.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">6️⃣ Explainability with SHAP</h3>
              <p>
                SHAP approximates Shapley values φ<sub>i</sub> explaining the marginal contribution of feature <code>i</code> to the model output:
              </p>
              <Equation value="phi_i = \sum_{S \subseteq F\setminus\{i\}} ( |S|! (|F|-|S|-1)! ) / |F|! * ( f(S\cup\{i\}) - f(S) )" />
            </div>
          </div>
        </section>

        {/* Reinforcement-Learning Pipeline */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
          <h2 className="text-3xl font-bold mb-6 flex items-center">
            <FaMicroscope className="mr-3 text-emerald-400" />
            Reinforcement Learning (Optional Training)
          </h2>
          <p className="text-gray-300 mb-6 leading-relaxed">
            The codebase contains a Proximal Policy Optimization (PPO) pipeline for fine-tuning the emotion classifier on
            domain-specific data. The agent maximises expected reward J(θ) = E<sub>τ∼π<sub>θ</sub></sub>[ R(τ) ]
            using the clipped objective:
          </p>
          <Equation value="L_CLIP(theta) = E_t[ min( r_t(theta) * Â_t , clip(r_t(theta), 1-epsilon, 1+epsilon) * Â_t ) ]" />
        </section>

        {/* Further Reading */}
        <section className="text-center text-sm text-gray-400">
          <p>
            View the annotated Jupyter notebooks for derivations in&nbsp;
            <a href="https://github.com/jadenfix/HumanVoiceAI" target="_blank" rel="noopener noreferrer" className="underline hover:text-white">the repository</a>.
          </p>
        </section>
      </div>
    </div>
  );
} 