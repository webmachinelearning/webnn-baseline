'use strict';

import {unary} from './lib/unary.js';

/**
 * Compute the hyperbolic tangent function of the input tensor.
 * The calculation follows the expression (exp(2 * x) - 1) / (exp(2 * x) + 1).
 * @param {Tensor} input
 * @return {Tensor}
 */
export function tanh(input) {
  return unary(input, (x) => (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1));
}
