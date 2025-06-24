'use strict';

import {mul, sub} from './binary.js';
import {blockwiseExpand} from './lib/broadcast.js';
import {validateQDQParams} from './lib/validate-input.js';

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
  validateQDQParams(input, scale, zeroPoint);

  const broadcastedScale = blockwiseExpand(scale, input.shape);
  const broadcastedZeroPoint = blockwiseExpand(zeroPoint, input.shape);
  return mul(sub(input, broadcastedZeroPoint), broadcastedScale);
}
