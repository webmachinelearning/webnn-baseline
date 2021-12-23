'use strict';

import {Tensor} from './tensor.js';

/**
 * Compute the rectified linear function of the input tensor.
 * @param {Tensor} input
 * @return {Tensor}
 */
export function relu(input) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < output.data.length; ++i) {
    output.data[i] = Math.max(0, input.data[i]);
  }
  return output;
}
