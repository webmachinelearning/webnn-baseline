'use strict';

import {unary} from './unary.js';

/**
 * Calculate the leaky version of rectified linear function on the input tensor element-wise.
 * @param {Tensor} input
 * @param {MLLeakyReluOptions} [options]
 * @return {Tensor}
 */
export function leakyRelu(input, options = {}) {
  const alpha = options.alpha !== undefined ? options.alpha : 0.01;
  return unary(input, (x) => Math.max(0, x) + alpha * Math.min(0, x));
}
