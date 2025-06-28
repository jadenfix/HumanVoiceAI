import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Page from '../page';

// Mock the audio recording functionality
const mockStartRecording = jest.fn();
const mockStopRecording = jest.fn();
const mockClearRecordedBlob = jest.fn();

// Mock the react-audio-voice-recorder module
jest.mock('react-audio-voice-recorder', () => ({
  useAudioRecorder: () => ({
    startRecording: mockStartRecording,
    stopRecording: mockStopRecording,
    recordingBlob: null,
    isRecording: false,
    recordingTime: 0,
  }),
  AudioRecorder: () => <div data-testid="audio-recorder">Audio Recorder</div>,
}));

// Mock the next/navigation module
const mockPush = jest.fn();
const mockRouter = {
  push: mockPush,
  pathname: '/',
  query: {},
  asPath: '/',
  events: {
    on: jest.fn(),
    off: jest.fn(),
  },
};

// Mock the next/navigation module
jest.mock('next/navigation', () => ({
  useRouter: () => mockRouter,
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
}));

describe('VoiceAIChat', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
    // Reset the DOM
    document.body.innerHTML = '';
  });

  it('renders the main container', () => {
    render(<Page />);
    expect(screen.getByTestId('voice-ai-chat')).toBeInTheDocument();
  });

  it('displays the header with title and description', () => {
    render(<Page />);
    
    // Check for the main heading
    const heading = screen.getByRole('heading', { level: 1, name: /Human Voice AI/i });
    expect(heading).toBeInTheDocument();
    
    // Check for the description
    const description = screen.getByText(/Record your voice and get AI-powered feedback on your speech patterns/i);
    expect(description).toBeInTheDocument();
  });

  it('shows the recording button with microphone icon', () => {
    render(<Page />);
    const recordButton = screen.getByRole('button', { name: /start recording/i });
    expect(recordButton).toBeInTheDocument();
    
    // Check if the microphone icon is present
    const micIcon = recordButton.querySelector('svg');
    expect(micIcon).toBeInTheDocument();
  });

  it('starts recording when the record button is clicked', async () => {
    render(<Page />);
    const recordButton = screen.getByRole('button', { name: /start recording/i });
    
    // Simulate click on the record button
    fireEvent.click(recordButton);
    
    // Check if startRecording was called
    await waitFor(() => {
      expect(mockStartRecording).toHaveBeenCalledTimes(1);
    });
  });

  it('displays the emotion selector with default options', () => {
    render(<Page />);
    
    // Check for the emotion selector label
    const emotionLabel = screen.getByLabelText(/Select Emotion/i);
    expect(emotionLabel).toBeInTheDocument();
    
    // Check for the default selected option
    const defaultOption = screen.getByDisplayValue('neutral');
    expect(defaultOption).toBeInTheDocument();
    
    // Check for all emotion options
    const emotionOptions = screen.getAllByRole('option');
    expect(emotionOptions.length).toBeGreaterThan(0);
  });

  it('displays the recording timer when recording is in progress', async () => {
    // Mock the useAudioRecorder hook to return isRecording as true
    jest.mock('react-audio-voice-recorder', () => ({
      useAudioRecorder: () => ({
        startRecording: mockStartRecording,
        stopRecording: mockStopRecording,
        recordingBlob: null,
        isRecording: true,
        recordingTime: 5, // 5 seconds
      }),
    }));
    
    render(<Page />);
    
    // Check if the recording timer is displayed
    const timer = screen.getByText(/00:05/);
    expect(timer).toBeInTheDocument();
    
    // Check if the stop button is visible
    const stopButton = screen.getByRole('button', { name: /stop recording/i });
    expect(stopButton).toBeInTheDocument();
  });

  it('shows the audio player when a recording is available', async () => {
    // Mock a recorded blob
    const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });
    
    // Mock the useAudioRecorder hook to return a recorded blob
    jest.mock('react-audio-voice-recorder', () => ({
      useAudioRecorder: () => ({
        startRecording: mockStartRecording,
        stopRecording: mockStopRecording,
        recordingBlob: mockBlob,
        isRecording: false,
        recordingTime: 0,
      }),
    }));
    
    render(<Page />);
    
    // Check if the audio player is displayed
    const audioPlayer = screen.getByTestId('audio-player');
    expect(audioPlayer).toBeInTheDocument();
    
    // Check if the analyze button is visible
    const analyzeButton = screen.getByRole('button', { name: /analyze recording/i });
    expect(analyzeButton).toBeInTheDocument();
  });
});
