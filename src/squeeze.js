'use strict';

import {reshape} from './reshape.js';

/**
 * Reduce the rank of a tensor by eliminating dimensions with size 1 of the tensor shape.
 * @param {Tensor} input
 * @param {MLSqueezeOptions} options
 * @return {Tensor}
 */
export function squeeze(input, options = {}) {
  let axes = options.axes;
  if (axes) {
    if (axes.length > input.rank) {
      throw new Error(`The length of axes ${axes.length} is bigger than input rank ${input.rank}.`);
    }

    for (const axis of axes) {
      if (axis < 0 || axis >= input.rank) {
        throw new Error(`The value of axes ${axis} is invalid.`);
      }
      if (options.axes && input.shape[axis] !== 1) {
        throw new Error(`The value ${input.shape[axis]} at axis ${axis} of input shape is not 1.`);
      }
    }
  } else {
    axes = new Array(input.rank).fill(0).map((_, i) => i);
  }

  const outputShape = input.shape.filter((dim, axis) => !(dim === 1 && axes.indexOf(axis) !== -1));
  const output = reshape(input, outputShape);
  return output;
}
