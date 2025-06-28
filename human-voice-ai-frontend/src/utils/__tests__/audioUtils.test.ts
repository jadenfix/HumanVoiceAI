import { formatDuration } from '../audioUtils';

describe('audioUtils', () => {
  describe('formatDuration', () => {
    it('formats seconds to MM:SS format', () => {
      expect(formatDuration(0)).toBe('00:00');
      expect(formatDuration(5)).toBe('00:05');
      expect(formatDuration(30)).toBe('00:30');
      expect(formatDuration(60)).toBe('01:00');
      expect(formatDuration(125)).toBe('02:05');
      expect(formatDuration(3600)).toBe('60:00');
    });

    it('handles negative numbers', () => {
      expect(formatDuration(-5)).toBe('00:00');
    });

    it('handles non-integer values', () => {
      expect(formatDuration(30.5)).toBe('00:30');
      expect(formatDuration(30.9)).toBe('00:30');
    });
  });
});
