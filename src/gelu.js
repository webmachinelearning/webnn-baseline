'use strict';

import {erfKernel, unary} from './unary.js';

/**
 * Compute the gaussian error linear unit function (GELU) of the input tensor.
 * The calculation follows the expression 0.5 * x * (1 + erf(x / sqrt(2))).
 * @param {Tensor} input
 * @return {Tensor}
 */
export function gelu(input) {
  return unary(input, (x) => 0.5 * x * (1 + erfKernel(x / Math.sqrt(2))));
}
