'use strict';

import {add, mul, max, min} from './binary.js';
import {Tensor} from './lib/tensor.js';

/**
 * Calculate the parametric version of rectified linear function (Parametric Relu) on the input
 * tensor element-wise. The calculation follows the expression max(0, x) + slope ∗ min(0, x).
 * @param {Tensor} x
 * @param {Tensor} slope
 * @return {Tensor}
 */
export function prelu(x, slope) {
  const zero = new Tensor([1], [0]);
  return add(max(zero, x), mul(slope, min(zero, x)));
}
