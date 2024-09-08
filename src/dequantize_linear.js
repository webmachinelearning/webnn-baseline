'use strict';

import {mul, sub} from './binary.js';

/**
 * Elementwise operator to scale a low precision integer (typically uint8 with a zero-point bias)
 * to floating point.
 * The calculation follows the expression (input - zeroPoint) * scale.
 * @param {Tensor} input
 * @param {Tensor} scale
 * @param {Tensor} zeroPoint
 * @return {Tensor}
 */
export function dequantizeLinear(input, scale, zeroPoint) {
  return mul(sub(input, zeroPoint), scale);
}
