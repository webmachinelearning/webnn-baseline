'use strict';

import {broadcast, getBroadcastShape} from './lib/broadcast.js';
import {Tensor} from '../src/lib/tensor.js';

/**
 * Expand any dimension of size 1 of the input tensor to a
 * larger size according to the new shape.
 * @param {Tensor} input
 * @param {Array} newShape
 * @return {Tensor}
 */
export function expand(input, newShape) {
  if (input.shape.length === 0) {
    const inputReshape = new Tensor([1], input.data);
    const outputShape = getBroadcastShape(inputReshape.shape, newShape);
    return broadcast(inputReshape, outputShape);
  } else {
    const inputReshape = new Tensor(input.shape, input.data);
    const outputShape = getBroadcastShape(inputReshape.shape, newShape);
    return broadcast(inputReshape, outputShape);
  }
}
