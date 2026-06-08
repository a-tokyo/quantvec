import { describe, expect, it } from 'vitest';
import { VERSION } from '../src/index';

describe('quantvec smoke', () => {
  it('exports a version string', () => {
    expect(typeof VERSION).toBe('string');
  });
});
