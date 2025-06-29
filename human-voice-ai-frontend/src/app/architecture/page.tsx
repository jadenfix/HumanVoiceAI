'use client';

import Link from 'next/link';
import { FaArrowLeft, FaCogs, FaNetworkWired } from 'react-icons/fa';

export default function ArchitecturePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-stone-800 text-gray-100">
      {/* Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-black/30" />
        <header className="relative z-10 container mx-auto px-6 py-6 flex items-center justify-between">
          <Link href="/" className="inline-flex items-center text-sm text-gray-300 hover:text-white">
            <FaArrowLeft className="mr-2" /> Back to App
          </Link>
          <h1 className="text-2xl font-semibold bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent">
            🏗️ System Architecture
          </h1>
        </header>
      </div>

      <div className="container mx-auto px-6 py-12 space-y-12">
        {/* High-Level Diagram */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-cyan-400/30 shadow-xl">
          <h2 className="text-3xl font-bold mb-6 flex items-center">
            <FaNetworkWired className="mr-3 text-cyan-400" />
            End-to-End Dataflow
          </h2>
          <p className="text-gray-300 mb-6 leading-relaxed">
            The diagram below illustrates the journey of an audio sample through every subsystem. Each numbered step is
            elaborated in subsequent sections.
          </p>
          <pre className="whitespace-pre overflow-x-auto text-sm md:text-base lg:text-lg bg-black/60 p-6 rounded-2xl font-mono text-teal-200 shadow-inner">
{`
 Browser (Next.js)          API Route              Streamlit Backend                 HuggingFace
 ─────────────────────────  ─────────────────────  ───────────────────────────────   ───────────
 1. Mic Recording  ───────▶  2. POST /process-audio ─▶  3. Save temp WAV          
                                           │        4. Feature Extraction (MFCC, spectral)
                                           │        5. wav2vec2 Embedding
                                           │        6. Emotion Classifier
                                           │        7. SHAP Explainer
                                           ▼
                              8. JSON Results ◀─────────────────────────────────────
 9. UI Visualisation ◀────────
`}
          </pre>
        </section>

        {/* Frontend Stack */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 grid md:grid-cols-2 gap-8 shadow-lg">
          <div>
            <h3 className="text-2xl font-semibold mb-4 flex items-center"><FaCogs className="mr-2 text-emerald-400" />Frontend</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-300">
              <li><strong>Next.js 14 App Router</strong> for file-system routing and API endpoints.</li>
              <li><strong>React 18</strong> functional components with hooks.</li>
              <li><strong>Tailwind&nbsp;CSS</strong> JIT classes for styling.</li>
              <li><strong>Framer-Motion</strong> for the animated recorder visualisation.</li>
              <li><strong>Web Audio API</strong> to capture and stream microphone input.</li>
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-semibold mb-4 flex items-center"><FaCogs className="mr-2 text-sky-400" />Backend</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-300">
              <li><strong>Streamlit 1.x</strong> app running on port 8501.</li>
              <li>HuggingFace <code>wav2vec2-lg-xlsr-en-speech-emotion-recognition</code> model (≈330 M parameters).</li>
              <li>Fallback heuristic classifier to guarantee predictions offline.</li>
              <li>Feature extraction with <code>librosa</code>, <code>numpy</code>, and custom DSP utilities.</li>
              <li>SHAP explainability for per-feature attributions.</li>
              <li>Optional RL fine-tuning pipeline using PPO in <code>human_voice_ai.rl</code>.</li>
            </ul>
          </div>
        </section>

        {/* Deployment & Ops */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 space-y-4 shadow-lg">
          <h2 className="text-3xl font-bold mb-4">DevOps &amp; Observability</h2>
          <ul className="list-disc list-inside space-y-2 text-gray-300">
            <li><strong>Shell scripts</strong> <code>deploy-local.sh</code> and <code>deploy-stable.sh</code> orchestrate the services, install dependencies, and ensure port availability.</li>
            <li><strong>Docker</strong> support via <code>Dockerfile</code> &amp; <code>docker-compose.yml</code> for containerised deployment.</li>
            <li><strong>GitHub Actions</strong> (planned) for CI, running unit &amp; integration tests.</li>
            <li><strong>Colored terminal output</strong> for quick status inspection during local runs.</li>
          </ul>
        </section>

        {/* Further Reading */}
        <section className="text-center text-sm text-gray-400">
          <p>
            Explore additional design documents in&nbsp;
            <a href="https://github.com/jadenfix/HumanVoiceAI" target="_blank" rel="noopener noreferrer" className="underline hover:text-white">the repository</a>.
          </p>
        </section>
      </div>
    </div>
  );
} 