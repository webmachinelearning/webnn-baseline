'use strict';

import {unary} from './unary.js';

/**
 * Compute the softplus function of the input tensor.
 * The calculation follows the expression ln(1 + exp(steepness * x)) / steepness.
 * @param {Tensor} input
 * @param {MLSoftplusOptions} [options]
 * @return {Tensor}
 */
export function softplus(input, {steepness=1} = {}) {
  return unary(
      input, (x) => Math.log(1 + Math.exp(steepness * x)) / steepness);
}
