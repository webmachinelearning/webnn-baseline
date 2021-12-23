'use strict';

import {Tensor} from './tensor.js';

/**
 * Compute the sigmoid function of the input tensor.
 * @param {Tensor} input
 * @return {Tensor}
 */
export function sigmoid(input) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < output.data.length; ++i) {
    output.data[i] = 1 / (Math.exp(-input.data[i]) + 1);
  }
  return output;
}
