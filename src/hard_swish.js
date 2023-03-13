'use strict';

import {unary} from './unary.js';

/**
 * Computes the nonlinear function on the input tensor element-wise.
 *     y = x * max(0, min(6, (x + 3))) / 6
 * @param {Tensor} input
 * @return {Tensor}
 */
export function hardSwish(input) {
  return unary(input, (x) => x * Math.max(0, Math.min(6, x + 3)) / 6);
}
