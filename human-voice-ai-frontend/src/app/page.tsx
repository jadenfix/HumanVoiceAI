'use client';

import { useState, useRef, useEffect } from 'react';
import { MicrophoneIcon, StopIcon, PlayIcon, PauseIcon } from '@heroicons/react/24/solid';

type Emotion = 'happy' | 'sad' | 'angry' | 'neutral' | 'excited';

interface Message {
  text: string;
  sender: 'user' | 'ai';
  emotion: Emotion;
  timestamp: Date;
}

export default function VoiceAIChat() {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentEmotion, setCurrentEmotion] = useState<Emotion>('neutral');
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Initialize audio context
  useEffect(() => {
    // Request microphone permission on component mount
    if (typeof window !== 'undefined') {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .catch(err => console.error('Microphone access denied:', err));
    }

    // Add welcome message
    setMessages([
      {
        text: 'Hello! I\'m your voice AI assistant. How can I help you today?',
        sender: 'ai',
        emotion: 'happy',
        timestamp: new Date()
      }
    ]);
    
    // Set default emotion
    setCurrentEmotion('neutral');
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsLoading(true);
    
    try {
      // Add user message immediately
      const userMessage: Message = {
        text: 'Processing your voice...',
        sender: 'user',
        emotion: currentEmotion,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, userMessage]);

      // Create form data to send the audio
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('emotion', currentEmotion);

      // Send to our API route
      const response = await fetch('/api/process-audio', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process audio');
      }

      const { text, emotion } = await response.json();

      // Update the user message with actual content
      const updatedUserMessage: Message = {
        text: 'Voice message',
        sender: 'user',
        emotion: currentEmotion,
        timestamp: new Date()
      };

      // Add AI response
      const aiMessage: Message = {
        text: text,
        sender: 'ai',
        emotion: emotion || 'neutral',
        timestamp: new Date()
      };

      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = updatedUserMessage; // Replace the loading message
        return [...updated, aiMessage];
      });
    } catch (error) {
      console.error('Error processing audio:', error);
      
      // Add error message
      const errorMessage: Message = {
        text: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'ai',
        emotion: 'sad',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const getEmotionColor = (emotion: Emotion) => {
    const colors = {
      happy: 'bg-yellow-100 text-yellow-800',
      sad: 'bg-blue-100 text-blue-800',
      angry: 'bg-red-100 text-red-800',
      neutral: 'bg-gray-100 text-gray-800',
      excited: 'bg-purple-100 text-purple-800',
    };
    return colors[emotion];
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Human Voice AI</h1>
          <p className="mt-1 text-sm text-gray-500">Emotion-aware voice assistant</p>
        </div>
      </header>

      <main className="max-w-3xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          {/* Chat messages */}
          <div className="px-4 py-5 sm:p-6 h-[60vh] overflow-y-auto">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.sender === 'ai' ? 'justify-start' : 'justify-end'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.sender === 'ai' ? 'bg-blue-100' : 'bg-green-100'
                    }`}
                  >
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-xs font-medium">
                        {message.sender === 'ai' ? 'AI' : 'You'}
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${getEmotionColor(
                          message.emotion
                        )}`}
                      >
                        {message.emotion}
                      </span>
                    </div>
                    <p className="text-sm text-gray-800">{message.text}</p>
                    <p className="text-xs text-gray-500 text-right mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 px-4 py-2 rounded-lg">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="bg-gray-50 px-4 py-4 sm:px-6 border-t border-gray-200">
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <div className="w-full sm:w-auto">
              <label htmlFor="emotion" className="block text-sm font-medium text-gray-700 mb-1">
                Select Emotion
              </label>
              <select
                id="emotion"
                value={currentEmotion}
                onChange={(e) => setCurrentEmotion(e.target.value as Emotion)}
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                disabled={isRecording}
              >
                <option value="happy">😊 Happy</option>
                <option value="sad">😢 Sad</option>
                <option value="angry">😠 Angry</option>
                <option value="neutral">😐 Neutral</option>
                <option value="excited">🤩 Excited</option>
              </select>
            </div>
            
            <div className="w-full sm:w-auto">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  <MicrophoneIcon className="h-5 w-5 mr-2" />
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  <StopIcon className="h-5 w-5 mr-2" />
                  Stop Recording
                </button>
              )}
            </div>

              <button
                onClick={togglePlayback}
                disabled={!audioRef.current}
                className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                  audioRef.current
                    ? 'bg-green-600 hover:bg-green-700 focus:ring-green-500'
                    : 'bg-gray-400 cursor-not-allowed'
                } focus:outline-none focus:ring-2 focus:ring-offset-2`}
              >
                {isPlaying ? (
                  <>
                    <PauseIcon className="h-5 w-5 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <PlayIcon className="h-5 w-5 mr-2" />
                    Play Last Response
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-white mt-8 border-t border-gray-200">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            &copy; {new Date().getFullYear()} Human Voice AI. All rights reserved.
          </p>
        </div>
      </footer>

      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
