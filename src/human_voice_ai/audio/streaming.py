"""
Real-time audio streaming and processing for emotion recognition.
"""

import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread, Event
from typing import Callable, Optional, Dict, Any
import time

class AudioStream:
    """Real-time audio streaming with callback support."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1600,  # 100ms at 16kHz
        channels: int = 1,
        device: Optional[int] = None,
        callback: Optional[Callable] = None
    ):
        """Initialize the audio stream.
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_size: Number of samples per chunk
            channels: Number of audio channels
            device: Audio device index (None for default)
            callback: Function to call with each audio chunk
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device = device
        self.callback = callback
        self.stream = None
        self.audio_queue = Queue()
        self.is_recording = False
        self.stop_event = Event()
        self.thread = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict, status: sd.CallbackFlags) -> None:
        """Callback function for audio stream."""
        if self.is_recording and not self.stop_event.is_set():
            self.audio_queue.put(indata.copy())
            if self.callback:
                self.callback(indata)

    def start(self) -> None:
        """Start the audio stream."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.stop_event.clear()
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            device=self.device
        )
        
        self.stream.start()
        
        # Start a thread to process the audio queue
        self.thread = Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the audio stream."""
        self.is_recording = False
        self.stop_event.set()
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _process_queue(self) -> None:
        """Process audio chunks from the queue."""
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                # Process audio chunks here if needed
                time.sleep(0.01)
            except Exception as e:
                print(f"Error processing audio: {e}")

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue.
        
        Args:
            timeout: Time to wait for a chunk in seconds
            
        Returns:
            Audio chunk as numpy array or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except Exception:
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class AudioProcessor:
    """Process audio chunks for emotion recognition."""
    
    def __init__(
        self,
        model: Any,
        sample_rate: int = 16000,
        chunk_size: int = 1600,
        device: str = 'cpu'
    ):
        """Initialize the audio processor.
        
        Args:
            model: Pre-trained emotion recognition model
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.buffer = np.zeros(chunk_size * 8, dtype=np.float32)  # 800ms buffer
        self.buffer_ptr = 0
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single audio chunk.
        
        Args:
            audio_chunk: Input audio chunk (n_samples, n_channels)
            
        Returns:
            Dictionary containing emotion predictions and metadata
        """
        # Convert to mono if needed
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # Update ring buffer
        chunk_len = len(audio_chunk)
        if chunk_len > len(self.buffer):
            self.buffer = np.zeros(chunk_len * 2, dtype=np.float32)
        
        start = self.buffer_ptr
        end = (self.buffer_ptr + chunk_len) % len(self.buffer)
        
        if end > start:
            self.buffer[start:end] = audio_chunk
        else:
            remaining = len(self.buffer) - start
            self.buffer[start:] = audio_chunk[:remaining]
            self.buffer[:end] = audio_chunk[remaining:]
        
        self.buffer_ptr = end
        
        # Get the last 1 second of audio
        if end > self.sample_rate:
            window = np.concatenate((self.buffer[end-self.sample_size:end], 
                                   self.buffer[:end-self.sample_size]))
        else:
            window = self.buffer[:self.sample_size]
        
        # Convert to tensor and process
        audio_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(audio_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        return {
            'emotion': pred_class,
            'probabilities': probs.cpu().numpy()[0],
            'audio': window
        }


def test_stream():
    """Test function for audio streaming."""
    def print_energy(audio_chunk):
        energy = np.mean(audio_chunk**2)
        print(f"Audio energy: {energy:.6f}")

    print("Starting audio stream test...")
    print("Press Ctrl+C to stop")
    
    try:
        with AudioStream(callback=print_energy) as stream:
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Test completed")


if __name__ == "__main__":
    test_stream()
