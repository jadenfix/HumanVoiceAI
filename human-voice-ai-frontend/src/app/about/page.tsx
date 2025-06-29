'use client';

import Link from 'next/link';
import { FaArrowLeft, FaGithub, FaLinkedin, FaEnvelope } from 'react-icons/fa';
import { motion } from 'framer-motion';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 text-gray-100">
      {/* Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-black/20"></div>
        <header className="relative z-10 container mx-auto px-6 py-6 flex items-center justify-between">
          <Link href="/" className="inline-flex items-center text-sm text-gray-300 hover:text-white">
            <FaArrowLeft className="mr-2" /> Back to App
          </Link>
          <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            👤 About the Creator
          </h1>
        </header>
      </div>

      <div className="container mx-auto px-6 py-12 space-y-12">
        {/* Bio Section */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
          <h2 className="text-3xl font-bold mb-4">Hi, I'm Jaden Fix 👋</h2>
          <p className="text-gray-300 leading-relaxed mb-6">
            I'm an engineer and researcher passionate about all things AI, ML, human-computer interaction, speech processing, and
            reinforcement learning. I built <span className="font-semibold">Voice AI</span> to showcase real-time emotion
            recognition with a highly interactive frontend and interpretable backend.
          </p>

          <div className="grid md:grid-cols-3 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-black/30 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold mb-2 text-white">🎓 Background</h3>
              <p className="text-sm text-gray-300 leading-relaxed">
                • M.S. Quant – Focus on ML &amp; DSP
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="bg-black/30 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold mb-2 text-white">🛠️ Current Work</h3>
              <p className="text-sm text-gray-300 leading-relaxed">
                • Building emotion-aware conversational agents
                <br />• Exploring RL for adaptive speech synthesis
                <br />• Consulting on AI product strategy
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="bg-black/30 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold mb-2 text-white">🚀 Mission</h3>
              <p className="text-sm text-gray-300 leading-relaxed">
                Democratise affective computing by making voice-first AI systems that are transparent, ethical, and delightful
                to use.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Contact Section */}
        <section className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 text-center">
          <h3 className="text-2xl font-bold mb-4">Let's connect</h3>
          <p className="text-gray-300 mb-6 max-w-xl mx-auto">
            I'm always excited to discuss AI, speech tech, or potential collaborations. Feel free to reach out!
          </p>
          <div className="flex items-center justify-center space-x-6">
            <a href="mailto:jadenfix123@gmail.com" className="hover:text-white transition-colors" title="Email">
              <FaEnvelope size={24} />
            </a>
            <a href="https://github.com/jadenfix" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors" title="GitHub">
              <FaGithub size={24} />
            </a>
            <a href="https://linkedin.com/in/jadenfix" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors" title="LinkedIn">
              <FaLinkedin size={24} />
            </a>
          </div>
        </section>
      </div>
    </div>
  );
} 