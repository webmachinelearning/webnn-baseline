'use strict';

import {unary} from './unary.js';

/**
 * Calculate a linear function y = alpha * x + beta on the input tensor.
 * @param {Tensor} input
 * @param {MLLinearOptions} [options]
 * @return {Tensor}
 */
export function linear(input, {alpha=1, beta=0} = {}) {
  return unary(input, (x) => alpha * x + beta);
}
