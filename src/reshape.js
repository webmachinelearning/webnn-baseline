'use strict';

import {Tensor, sizeOfShape} from './tensor.js';

/**
 *
 * @param {Tensor} input
 * @param {Array} newShape
 * @return {Tensor}
 */
export function reshape(input, newShape) {
  let minusOneAxis;
  let elements = 1;
  for (let i = 0; i < newShape.length; ++i) {
    if (newShape[i] === -1) {
      minusOneAxis = i;
    } else if (newShape[i] > 0) {
      elements *= newShape[i];
    } else {
      throw new Error('The new shape is invalid.');
    }
  }
  const outputShape = newShape.slice();
  if (minusOneAxis !== undefined) {
    outputShape[minusOneAxis] = Math.round(sizeOfShape(input.shape) / elements);
  }
  if (sizeOfShape(input.shape) !== sizeOfShape(outputShape)) {
    throw new Error('The new shape is invalid.');
  }
  const output = new Tensor(outputShape, input.data);
  return output;
}
