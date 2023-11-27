'use strict';

import {broadcast, getBroadcastShape} from './lib/broadcast.js';
import {Scalar} from '../src/lib/tensor.js';

/**
 * Expand any dimension of size 1 of the input tensor to a
 * larger size according to the new shape.
 * @param {Tensor} input
 * @param {Array} newShape
 * @return {Tensor}
 */


export function expand(input, newShape) {
  const inputReshape = input.shape.length === 0 ? new Scalar(input.data) : input;
  const outputShape = getBroadcastShape(inputReshape.shape, newShape);
  return broadcast(inputReshape, outputShape);
}
