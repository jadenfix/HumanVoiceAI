'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaMicrophone, FaStop, FaPlay, FaChartLine, FaBrain, FaWaveSquare } from 'react-icons/fa';

interface EmotionResult {
  score: number;
  confidence: number;
  label: string;
  timestamp: number;
}

export default function VoiceAI() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<EmotionResult | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionResult[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const emotionMap = {
    1: { label: 'Very Negative', emoji: '😠', color: 'from-red-500 to-red-600' },
    2: { label: 'Slightly Negative', emoji: '😔', color: 'from-orange-500 to-orange-600' },
    3: { label: 'Neutral', emoji: '😐', color: 'from-gray-500 to-gray-600' },
    4: { label: 'Slightly Positive', emoji: '🙂', color: 'from-green-500 to-green-600' },
    5: { label: 'Very Positive', emoji: '😊', color: 'from-emerald-500 to-emerald-600' }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      
      source.connect(analyser);
      analyser.fftSize = 256;
      
      mediaRecorderRef.current = mediaRecorder;
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      
      const audioChunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await processAudio(audioBlob);
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start audio level monitoring
      const updateAudioLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
          setAudioLevel(average / 255);
        }
      };
      
      // Start timer and audio level updates
      intervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 0.1);
        updateAudioLevel();
      }, 100);
      
      // Auto-stop after 3 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current && isRecording) {
          stopRecording();
        }
      }, 3000);
      
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setAudioLevel(0);
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      
      const response = await fetch('/api/process-audio', {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      
      if (result.success) {
        const emotionResult: EmotionResult = {
          score: result.emotion.score,
          confidence: result.emotion.confidence,
          label: result.emotion.label,
          timestamp: Date.now()
        };
        
        setCurrentEmotion(emotionResult);
        setEmotionHistory(prev => [...prev.slice(-9), emotionResult]);
      }
    } catch (error) {
      console.error('Error processing audio:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const clearHistory = () => {
    setEmotionHistory([]);
    setCurrentEmotion(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative z-10 container mx-auto px-6 py-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <div className="flex items-center justify-center mb-6">
              <FaBrain className="text-6xl text-blue-400 mr-4" />
              <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Voice AI
              </h1>
            </div>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Advanced emotion detection using state-of-the-art machine learning models
            </p>
          </motion.div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          
          {/* Recording Section */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20"
          >
            <div className="text-center">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center justify-center">
                <FaMicrophone className="mr-3 text-blue-400" />
                Voice Recording
              </h2>
              
              {/* Audio Visualizer */}
              <div className="mb-8">
                <div className="h-32 bg-black/30 rounded-2xl flex items-center justify-center mb-4">
                  <div className="flex items-end space-x-1">
                    {Array.from({ length: 20 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-2 bg-gradient-to-t from-blue-500 to-purple-500 rounded-full"
                        animate={{
                          height: isRecording ? `${Math.random() * audioLevel * 100 + 10}px` : '10px'
                        }}
                        transition={{ duration: 0.1 }}
                      />
                    ))}
                  </div>
                </div>
                
                {isRecording && (
                  <div className="text-white text-sm">
                    Recording: {recordingTime.toFixed(1)}s / 3.0s
                  </div>
                )}
              </div>

              {/* Recording Controls */}
              <div className="space-y-4">
                <AnimatePresence>
                  {!isRecording && !isProcessing && (
                    <motion.button
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                      onClick={startRecording}
                      className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-2xl shadow-lg hover:shadow-xl transition-shadow"
                    >
                      <FaMicrophone />
                    </motion.button>
                  )}
                  
                  {isRecording && (
                    <motion.button
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                      onClick={stopRecording}
                      className="w-24 h-24 bg-gradient-to-r from-red-500 to-pink-500 rounded-full flex items-center justify-center text-white text-2xl shadow-lg hover:shadow-xl transition-shadow animate-pulse"
                    >
                      <FaStop />
                    </motion.button>
                  )}
                  
                  {isProcessing && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                      className="w-24 h-24 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full flex items-center justify-center text-white text-2xl shadow-lg"
                    >
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                    </motion.div>
                  )}
                </AnimatePresence>
                
                <p className="text-gray-300 text-sm">
                  {!isRecording && !isProcessing && 'Click to record 3 seconds of audio'}
                  {isRecording && 'Recording... Speak now!'}
                  {isProcessing && 'Analyzing emotion...'}
                </p>
              </div>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20"
          >
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
              <FaChartLine className="mr-3 text-green-400" />
              Emotion Analysis
            </h2>
            
            {currentEmotion ? (
              <div className="space-y-6">
                {/* Current Emotion */}
                <div className={`bg-gradient-to-r ${emotionMap[currentEmotion.score as keyof typeof emotionMap].color} rounded-2xl p-6 text-white text-center`}>
                  <div className="text-6xl mb-4">
                    {emotionMap[currentEmotion.score as keyof typeof emotionMap].emoji}
                  </div>
                  <div className="text-2xl font-bold mb-2">
                    {emotionMap[currentEmotion.score as keyof typeof emotionMap].label}
                  </div>
                  <div className="text-4xl font-bold mb-2">
                    {currentEmotion.score}/5
                  </div>
                  <div className="text-sm opacity-90">
                    Confidence: {(currentEmotion.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Confidence Meter */}
                <div className="bg-black/30 rounded-2xl p-4">
                  <div className="text-white text-sm mb-2">Confidence Level</div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${currentEmotion.confidence * 100}%` }}
                      transition={{ duration: 0.8 }}
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full"
                    />
                  </div>
                  <div className="text-right text-white text-xs mt-1">
                    {(currentEmotion.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-12">
                <FaWaveSquare className="text-6xl mb-4 mx-auto opacity-50" />
                <p className="text-lg">Record audio to see emotion analysis</p>
              </div>
            )}
          </motion.div>
        </div>

        {/* History Section */}
        {emotionHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-12 bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20"
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold text-white flex items-center">
                <FaChartLine className="mr-3 text-purple-400" />
                Emotion Timeline
              </h2>
              <button
                onClick={clearHistory}
                className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-colors"
              >
                Clear History
              </button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-10 gap-4">
              {emotionHistory.map((emotion, index) => (
                <motion.div
                  key={emotion.timestamp}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className={`bg-gradient-to-r ${emotionMap[emotion.score as keyof typeof emotionMap].color} rounded-xl p-4 text-white text-center`}
                >
                  <div className="text-2xl mb-1">
                    {emotionMap[emotion.score as keyof typeof emotionMap].emoji}
                  </div>
                  <div className="text-lg font-bold">
                    {emotion.score}
                  </div>
                  <div className="text-xs opacity-80">
                    {(emotion.confidence * 100).toFixed(0)}%
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
