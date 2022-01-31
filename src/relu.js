'use strict';

import {unary} from './lib/unary.js';

/**
 * Compute the rectified linear function of the input tensor.
 * @param {Tensor} input
 * @return {Tensor}
 */
export function relu(input) {
  return unary(input, (x) => Math.max(0, x));
}
