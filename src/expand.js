'use strict';

import {broadcast} from './lib/broadcast.js';

/**
 * Expand any dimension of size 1 of the input tensor to a
 * larger size according to the new shape.
 * @param {Tensor} input
 * @param {Array} newShape
 * @return {Tensor}
 */
export function expand(input, newShape) {
  return broadcast(input, newShape);
}
