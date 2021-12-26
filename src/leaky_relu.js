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
  for (let i = 0; i < input.size; ++i) {
    const x = input.getValueByIndex(i);
    const y = Math.max(0, x) + alpha * Math.min(0, x);
    output.setValueByIndex(i, y);
  }
  return output;
}
