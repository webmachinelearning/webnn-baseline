'use strict';

import {unary} from './unary.js';

/**
 * Calculate the non-smooth function used in place of a sigmoid function on the input tensor.
 * @param {Tensor} input
 * @param {MLHardSigmoidOptions} [options]
 * @return {Tensor}
 */
export function hardSigmoid(input, {alpha=0.2, beta=0.5} = {}) {
  // max(min(alpha * x + beta, 1), 0)
  return unary(input, (x) => Math.max(Math.min(alpha * x + beta, 1), 0));
}
