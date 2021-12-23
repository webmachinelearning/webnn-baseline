'use strict';

import {Tensor} from './tensor.js';

/**
 * Calculate the leaky version of rectified linear function on the input tensor element-wise.
 * @param {Tensor} input
 * @param {MLLeakyReluOptions} [options]
 * @return {Tensor}
 */
export function leakyRelu(input, options = {}) {
  const alpha = options.alpha ? options.alpha : 0.01;
  const output = new Tensor(input.shape);
  for (let i = 0; i < output.data.length; ++i) {
    const x = input.data[i];
    output.data[i] = Math.max(0, x) + alpha * Math.min(0, x);
  }
  return output;
}
