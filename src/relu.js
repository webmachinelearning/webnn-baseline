'use strict';

import {Tensor} from './tensor.js';

/**
 * Compute the rectified linear function of the input tensor.
 * @param {Tensor} input
 * @return {Tensor}
 */
export function relu(input) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < output.size; ++i) {
    const x = input.getValueByIndex(i);
    const y = Math.max(0, x);
    output.setValueByIndex(i, y);
  }
  return output;
}
