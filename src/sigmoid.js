'use strict';

import {unary} from './unary.js';

/**
 * Compute the sigmoid function of the input tensor.
 * The calculation follows the expression 1 / (exp(-x) + 1).
 * @param {Tensor} input
 * @return {Tensor}
 */
export function sigmoid(input) {
  return unary(input, (x) => 1 / (Math.exp(-x) + 1));
}
