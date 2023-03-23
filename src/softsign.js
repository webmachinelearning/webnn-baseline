'use strict';

import {unary} from './unary.js';

/**
 * Compute the softsign function of the input tensor.
 * The calculation follows the expression x / (1 + |x|).
 * @param {Tensor} input
 * @return {Tensor}
 */
export function softsign(input) {
  return unary(input, (x) => x / (1 + Math.abs(x)));
}
