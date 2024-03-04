'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateSqueezeParams} from './lib/validate-input.js';

/**
 * Alter the shape of a tensor to a new shape.
 * @param {Tensor} input
 * @param {Array} newShape
 * @return {Tensor}
 */
export function reshape(input, newShape) {
  let minusOneAxis;
  let elements = 1;
  for (let i = 0; i < newShape.length; ++i) {
    if (newShape[i] === null) {
      minusOneAxis = i;
    } else if (newShape[i] > 0) {
      elements *= newShape[i];
    } else {
      throw new Error(`The value ${newShape[i]} at axis ${i} of new shape is invalid.`);
    }
  }
  const outputShape = newShape.slice();
  if (minusOneAxis !== undefined) {
    outputShape[minusOneAxis] = Math.round(sizeOfShape(input.shape) / elements);
  }
  if (sizeOfShape(input.shape) !== sizeOfShape(outputShape)) {
    throw new Error(`The element size of new shape ${sizeOfShape(outputShape)} is not equal to
        element size of old shape ${sizeOfShape(input.shape)} invalid.`);
  }
  const output = new Tensor(outputShape, input.data);
  return output;
}

/**
 * Reduce the rank of a tensor by eliminating dimensions with size 1 of the tensor shape.
 * @param {Tensor} input
 * @param {MLSqueezeOptions} options
 * @return {Tensor}
 */
export function squeeze(input, {axes} = {}) {
  validateSqueezeParams(...arguments);
  const inputAxes = axes ?? Array.from({length: input.rank}, (_, i) => i);
  const outputShape = input.shape.filter((dim, axis) =>
    !(dim === 1 && inputAxes.indexOf(axis) !== -1));
  const output = reshape(input, outputShape);
  return output;
}
