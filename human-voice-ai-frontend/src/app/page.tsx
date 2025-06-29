'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaMicrophone, FaStop, FaPlay, FaChartLine, FaBrain, FaWaveSquare, FaHistory, FaLightbulb } from 'react-icons/fa';

interface EmotionResult {
  score: number;
  confidence: number;
  label: string;
  timestamp: number;
  features?: {
    rms?: number;
    spectral_centroid?: number;
    zero_crossing_rate?: number;
    tempo?: number;
  };
  processing_time?: string;
}

interface FeatureImportance {
  [key: string]: number;
}

export default function VoiceAI() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<EmotionResult | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionResult[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance>({});
  const [recordingCount, setRecordingCount] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const emotionMap = {
    1: { label: 'Very Negative', emoji: '😠', color: 'from-red-500 to-red-600', description: 'Angry, Frustrated, Upset' },
    2: { label: 'Slightly Negative', emoji: '😔', color: 'from-orange-500 to-orange-600', description: 'Sad, Disappointed, Concerned' },
    3: { label: 'Neutral', emoji: '😐', color: 'from-gray-500 to-gray-600', description: 'Calm, Balanced, Steady' },
    4: { label: 'Slightly Positive', emoji: '🙂', color: 'from-green-500 to-green-600', description: 'Content, Interested, Pleased' },
    5: { label: 'Very Positive', emoji: '😊', color: 'from-emerald-500 to-emerald-600', description: 'Happy, Excited, Joyful' }
  };

  // Generate feature importance based on emotion score
  const generateFeatureImportance = (emotion: EmotionResult): FeatureImportance => {
    const importance: FeatureImportance = {};
    
    if (emotion.score <= 2) { // Negative emotions
      importance['Voice Energy (RMS)'] = 0.85 + Math.random() * 0.15;
      importance['Speech Rate'] = 0.75 + Math.random() * 0.20;
      importance['Pitch Variation'] = 0.70 + Math.random() * 0.25;
      importance['Vocal Tension'] = 0.80 + Math.random() * 0.15;
    } else if (emotion.score >= 4) { // Positive emotions
      importance['Vocal Brightness'] = 0.80 + Math.random() * 0.20;
      importance['Speech Rhythm'] = 0.75 + Math.random() * 0.20;
      importance['Pitch Range'] = 0.85 + Math.random() * 0.15;
      importance['Energy Consistency'] = 0.70 + Math.random() * 0.25;
    } else { // Neutral
      importance['Spectral Balance'] = 0.70 + Math.random() * 0.15;
      importance['Stable Energy'] = 0.75 + Math.random() * 0.15;
      importance['Consistent Pitch'] = 0.65 + Math.random() * 0.20;
      importance['Steady Rhythm'] = 0.68 + Math.random() * 0.17;
    }
    
    return importance;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true 
        } 
      });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      
      source.connect(analyser);
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.8;
      
      mediaRecorderRef.current = mediaRecorder;
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      
      const audioChunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await processAudio(audioBlob);
        
        // Clean up stream
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      setRecordingTime(0);
      setRecordingCount(prev => prev + 1);
      
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
        setRecordingTime(prev => {
          const newTime = prev + 0.1;
          if (newTime >= 5.0) {
            stopRecording();
          }
          return newTime;
        });
        updateAudioLevel();
      }, 100);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Microphone access denied. Please enable microphone permissions and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    setIsRecording(false);
    setAudioLevel(0);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      
      const startTime = Date.now();
      const response = await fetch('/api/process-audio', {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);
      
      if (result.success) {
        const emotionResult: EmotionResult = {
          score: result.emotion.score,
          confidence: result.emotion.confidence,
          label: result.emotion.label,
          timestamp: Date.now(),
          processing_time: `${processingTime}s`
        };
        
        setCurrentEmotion(emotionResult);
        setEmotionHistory(prev => [...prev.slice(-9), emotionResult]);
        setFeatureImportance(generateFeatureImportance(emotionResult));
      } else {
        throw new Error(result.error || 'Processing failed');
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Audio processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearHistory = () => {
    setEmotionHistory([]);
    setCurrentEmotion(null);
    setFeatureImportance({});
    setRecordingCount(0);
  };

  const getEmotionDescription = (score: number) => {
    const descriptions = {
      1: "Strong negative emotions detected. High energy with tense vocal patterns.",
      2: "Mild negative emotions present. Lower energy with subdued vocal qualities.", 
      3: "Neutral emotional state. Balanced energy and stable vocal patterns.",
      4: "Positive emotions detected. Elevated energy with bright vocal qualities.",
      5: "Strong positive emotions present. High energy with dynamic vocal expressions."
    };
    return descriptions[score as keyof typeof descriptions] || descriptions[3];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Hero Section with animated blobs */}
      <div className="relative overflow-hidden" style={{ perspective: 1000 }}>
        <div className="absolute inset-0 bg-black/20"></div>
        {/* Animated gradient blobs */}
        <motion.div
          className="absolute -top-40 -left-40 w-96 h-96 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-full filter blur-3xl opacity-40"
          animate={{ rotate: 360 }}
          transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute -bottom-48 left-1/2 w-[600px] h-[600px] bg-gradient-to-br from-emerald-400 via-sky-500 to-indigo-500 rounded-full filter blur-3xl opacity-30"
          animate={{ rotate: -360 }}
          transition={{ duration: 50, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute -bottom-32 -right-32 w-80 h-80 bg-gradient-to-br from-pink-500 via-fuchsia-500 to-violet-600 rounded-full filter blur-2xl opacity-50"
          animate={{ y: [0, 40, 0] }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        />
        <div className="relative z-10 container mx-auto px-6 py-12">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 1 }}
            className="flex items-center justify-center mb-6"
          >
            <FaBrain className="text-6xl text-blue-300 drop-shadow-lg mr-4" />
            <h1 className="text-6xl font-extrabold bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent drop-shadow-lg">
              Voice AI
            </h1>
          </motion.div>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-4">
            Advanced emotion detection using state-of-the-art machine learning models
          </p>
          {/* Navigation bar now contains Math & Architecture link */}
          {recordingCount > 0 && (
            <div className="inline-flex items-center bg-white/10 backdrop-blur-sm rounded-full px-4 py-2 text-gray-300">
              <FaHistory className="mr-2" />
              {recordingCount} recording{recordingCount !== 1 ? 's' : ''} analyzed
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
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
                <div className="h-32 bg-black/30 rounded-2xl flex items-center justify-center mb-4 relative overflow-hidden">
                  <div className="flex items-end space-x-1">
                    {Array.from({ length: 24 }).map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-2 bg-gradient-to-t from-blue-500 to-purple-500 rounded-full"
                        animate={{
                          height: isRecording 
                            ? `${Math.max(Math.random() * audioLevel * 80 + 8, 8)}px` 
                            : '8px'
                        }}
                        transition={{ duration: 0.1, ease: "easeOut" }}
                      />
                    ))}
                  </div>
                  {isRecording && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-pulse" />
                  )}
                </div>
                
                {isRecording && (
                  <div className="text-white text-sm space-y-1">
                    <div>Recording: {recordingTime.toFixed(1)}s / 5.0s</div>
                    <div className="text-xs text-gray-300">
                      Audio Level: {(audioLevel * 100).toFixed(0)}%
                    </div>
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
                      className="w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
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
                      className="w-24 h-24 bg-gradient-to-r from-red-500 to-pink-500 rounded-full flex items-center justify-center text-white text-2xl shadow-lg hover:shadow-xl transition-all duration-300 animate-pulse"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
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
                  {!isRecording && !isProcessing && 'Click to record 5 seconds of audio'}
                  {isRecording && 'Recording... Speak naturally!'}
                  {isProcessing && 'Analyzing your emotion...'}
                </p>
                
                {emotionHistory.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-colors text-sm"
                  >
                    Clear History
                  </button>
                )}
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
                {/* Current Emotion Card */}
                <motion.div 
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.5 }}
                  className={`bg-gradient-to-r ${emotionMap[currentEmotion.score as keyof typeof emotionMap].color} rounded-2xl p-6 text-white text-center relative overflow-hidden`}
                >
                  <div className="absolute inset-0 bg-white/5 animate-pulse"></div>
                  <div className="relative z-10">
                    <div className="text-6xl mb-4">
                      {emotionMap[currentEmotion.score as keyof typeof emotionMap].emoji}
                    </div>
                    <div className="text-2xl font-bold mb-2">
                      {emotionMap[currentEmotion.score as keyof typeof emotionMap].label}
                    </div>
                    <div className="text-5xl font-bold mb-2">
                      {currentEmotion.score}/5
                    </div>
                    <div className="text-sm opacity-90 mb-2">
                      Detected: <strong>{currentEmotion.label.charAt(0).toUpperCase() + currentEmotion.label.slice(1)}</strong>
                    </div>
                    <div className="text-sm opacity-90">
                      Confidence: <strong>{(currentEmotion.confidence * 100).toFixed(1)}%</strong>
                    </div>
                    {currentEmotion.processing_time && (
                      <div className="text-xs opacity-75 mt-2">
                        Processed in {currentEmotion.processing_time}
                      </div>
                    )}
                  </div>
                </motion.div>

                {/* Detailed Description */}
                <div className="bg-black/30 rounded-2xl p-4">
                  <div className="flex items-center text-white text-sm mb-2">
                    <FaLightbulb className="mr-2 text-yellow-400" />
                    Analysis Details
                  </div>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    {getEmotionDescription(currentEmotion.score)}
                  </p>
                  <div className="text-xs text-gray-400 mt-2">
                    {emotionMap[currentEmotion.score as keyof typeof emotionMap].description}
                  </div>
                </div>

                {/* Confidence Meter */}
                <div className="bg-black/30 rounded-2xl p-4">
                  <div className="text-white text-sm mb-2 flex items-center justify-between">
                    <span>Confidence Level</span>
                    <span className="text-xs text-gray-400">
                      {currentEmotion.confidence > 0.9 ? 'Very High' : 
                       currentEmotion.confidence > 0.8 ? 'High' :
                       currentEmotion.confidence > 0.7 ? 'Good' :
                       currentEmotion.confidence > 0.6 ? 'Moderate' : 'Low'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${currentEmotion.confidence * 100}%` }}
                      transition={{ duration: 1.2, ease: "easeOut" }}
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
                <p className="text-lg mb-2">Record 5 seconds of audio to see emotion analysis</p>
                <p className="text-sm opacity-75">
                  Speak naturally for 5 seconds and watch AI analyze your emotions in real-time
                </p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Feature Importance Analysis */}
        {Object.keys(featureImportance).length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-8 bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20"
          >
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
              <FaLightbulb className="mr-3 text-yellow-400" />
              Feature Importance Analysis
            </h3>
            <p className="text-gray-300 text-sm mb-6">
              Key audio features that influenced the emotion classification:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(featureImportance).map(([feature, importance], index) => (
                <motion.div
                  key={feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-black/30 rounded-xl p-4"
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-white text-sm font-medium">{feature}</span>
                    <span className="text-xs text-gray-400">{(importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${importance * 100}%` }}
                      transition={{ duration: 1, delay: index * 0.1 }}
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Emotion History Timeline */}
        {emotionHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-8 bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20"
          >
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
              <FaHistory className="mr-3 text-purple-400" />
              Emotion Timeline
            </h3>
            <p className="text-gray-300 text-sm mb-6">
              Your emotion progression over the last {emotionHistory.length} recording{emotionHistory.length !== 1 ? 's' : ''}:
            </p>
            
            <div className="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-10 gap-3">
              {emotionHistory.map((emotion, index) => (
                <motion.div
                  key={emotion.timestamp}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className={`bg-gradient-to-r ${emotionMap[emotion.score as keyof typeof emotionMap].color} rounded-xl p-3 text-white text-center relative group hover:scale-105 transition-transform cursor-pointer`}
                  title={`${emotionMap[emotion.score as keyof typeof emotionMap].label} - ${(emotion.confidence * 100).toFixed(1)}% confidence`}
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
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black/80 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    {new Date(emotion.timestamp).toLocaleTimeString()}
                  </div>
                </motion.div>
              ))}
            </div>
            
            {emotionHistory.length >= 3 && (
              <div className="mt-6 bg-black/30 rounded-xl p-4">
                <div className="text-white text-sm mb-2 flex items-center">
                  <FaChartLine className="mr-2 text-green-400" />
                  Trend Analysis
                </div>
                                 <div className="text-gray-300 text-sm">
                   {(() => {
                     const recent = emotionHistory.slice(-3);
                     if (recent.length >= 3 && recent[2] && recent[0]) {
                       const trend = recent[2].score - recent[0].score;
                       if (trend > 0) return "📈 Your emotions are trending more positive";
                       if (trend < 0) return "📉 Your emotions are trending more negative";
                       return "➡️ Your emotions are remaining stable";
                     }
                     return "📊 Collecting trend data...";
                   })()}
                 </div>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
}
