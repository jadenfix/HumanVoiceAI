/**
 * Formats a duration in seconds to MM:SS format
 * @param seconds - The duration in seconds
 * @returns Formatted time string in MM:SS format
 */
export const formatDuration = (seconds: number): string => {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;
  
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

/**
 * Converts a Blob to a base64 string
 * @param blob - The Blob to convert
 * @returns A promise that resolves to a base64 string
 */
export const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        const base64 = reader.result.split(',')[1] ?? '';
        resolve(base64); // Remove the data URL prefix
      } else {
        reject(new Error('Failed to read blob'));
      }
    };
    reader.onerror = () => {
      reject(new Error('Failed to read blob'));
    };
    reader.readAsDataURL(blob);
  });
};

/**
 * Creates an audio element from a Blob
 * @param blob - The audio Blob
 * @returns An HTMLAudioElement
 */
export const createAudioElement = (blob: Blob): HTMLAudioElement => {
  const audioUrl = URL.createObjectURL(blob);
  const audio = new Audio(audioUrl);
  return audio;
};
