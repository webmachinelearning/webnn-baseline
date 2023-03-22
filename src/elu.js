'use strict';

import {unary} from './unary.js';

/**
 * Calculate the exponential linear unit function on the input tensor element-wise.
 * The calculation follows the expression max(0, x) + alpha * (exp(min(0, x)) - 1).
 * @param {Tensor} input
 * @param {MLEluOptions} [options]
 * @return {Tensor}
 */
export function elu(input, {alpha=1} = {}) {
  return unary(
      input,
      (x) => Math.max(0, x) + alpha * (Math.exp(Math.min(0, x)) - 1),
  );
}
