'use strict';

import {Tensor} from './tensor.js';

/**
 * Compute the sigmoid function of the input tensor.
 * @param {Tensor} input
 * @return {Tensor}
 */
export function sigmoid(input) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < input.size; ++i) {
    const x = input.getValueByIndex(i);
    const y = 1 / (Math.exp(-x) + 1);
    output.setValueByIndex(i, y);
  }
  return output;
}
