'use client';

import Link from 'next/link';
import { FaArrowLeft, FaBrain, FaCube, FaWaveSquare } from 'react-icons/fa';
import { motion } from 'framer-motion';

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 text-gray-100">
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-black/20"></div>
        <header className="relative z-10 container mx-auto px-6 py-6 flex items-center justify-between">
          <Link href="/" className="inline-flex items-center text-sm text-gray-300 hover:text-white">
            <FaArrowLeft className="mr-2" /> Back to App
          </Link>
          <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            📚 Voice AI – Math & Architecture
          </h1>
        </header>
      </div>

      <div className="container mx-auto px-6 py-12 space-y-12">
        {/* System Architecture */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
          <h2 className="text-3xl font-bold mb-4 flex items-center">
            <FaCube className="mr-3 text-emerald-400" />
            System Architecture
          </h2>
          <p className="text-gray-300 mb-6 leading-relaxed">
            The Voice AI pipeline is composed of a modern React&nbsp;/ Next.js frontend and a Streamlit backend that
            leverages pre-trained HuggingFace models. Audio is recorded in the browser, sent to the backend, processed,
            and the results are streamed back to the UI.
          </p>

          <div className="grid md:grid-cols-2 gap-6 text-sm">
            <div className="bg-black/30 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3">Frontend – Next.js 14</h3>
              <ul className="list-disc list-inside space-y-1 text-gray-300">
                <li>Web Audio API for microphone capture</li>
                <li>Framer Motion for realtime animations</li>
                <li>Tailwind CSS for rapid styling</li>
                <li>REST API communication (<code>/api/process-audio</code>)</li>
              </ul>
            </div>
            <div className="bg-black/30 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3">Backend – Streamlit</h3>
              <ul className="list-disc list-inside space-y-1 text-gray-300">
                <li>File-based 16 kHz mono audio ingestion</li>
                <li>Feature extraction (MFCC, spectral, tempo)</li>
                <li>HuggingFace <code>wav2vec2</code> emotion model</li>
                <li>Fallback rule-based heuristics for robustness</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Math Behind the Models */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
          <h2 className="text-3xl font-bold mb-4 flex items-center">
            <FaWaveSquare className="mr-3 text-sky-400" />
            Math Behind Emotion Detection
          </h2>

          <div className="space-y-6 text-gray-300 leading-relaxed">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">1️⃣ Signal Pre-processing</h3>
              <p>
                The raw waveform <code>x[n]</code> is normalised and resampled to 16 kHz mono. A Hann window is applied with
                25 ms frames and 10 ms hop to create short-time segments.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">2️⃣ Feature Extraction (MFCC)</h3>
              <p>
                For each frame the magnitude spectrum <code>|X(k)|</code> is passed through a Mel filter-bank. Taking the
                discrete cosine transform (DCT) of the log energies yields the Mel-Frequency Cepstral Coefficients:
              </p>
              <p className="bg-black/40 p-4 rounded text-sm font-mono overflow-x-auto">
                c<sub>m</sub> = Σ<sub>k=1</sub><sup>K</sup> log|X(k)| · cos[ m (k−0.5) π ⁄ K ]
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">3️⃣ Spectral Descriptors</h3>
              <p>
                Additional descriptors such as spectral centroid, roll-off and zero-crossing rate capture timbral and
                temporal characteristics correlated with emotional prosody.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">4️⃣ Transformer Embedding</h3>
              <p>
                The processed frames are fed into a pre-trained <span className="font-semibold">wav2vec 2.0</span> model which
                outputs a contextual embedding <code>h</code>. A linear classification head maps <code>h</code> to emotion logits
                followed by softmax to obtain class probabilities.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">5️⃣ Confidence &amp; Interpretation</h3>
              <p>
                The highest probability defines the predicted emotion <code>ŷ</code>. SHAP values are used to approximate the
                contribution of each feature to <code>ŷ</code>, enabling the feature-importance visualisations you see in the app.
              </p>
            </div>
          </div>
        </section>

        {/* Further Reading */}
        <section className="text-center text-sm text-gray-400">
          <p>
            Want to dig deeper? Check out the source on&nbsp;
            <a href="https://github.com/jadenfix/HumanVoiceAI" target="_blank" rel="noopener noreferrer" className="underline hover:text-white">GitHub</a>.
          </p>
        </section>
      </div>
    </div>
  );
} 