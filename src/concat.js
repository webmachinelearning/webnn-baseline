'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';

/**
 * Concatenates the input tensors along a given axis.
 * @param {Array.<Tensor>} inputs
 * @param {Number} axis
 * @return {Tensor}
 */
export function concat(inputs, axis) {
  const rank = inputs[0].rank;
  if (!Number.isInteger(axis)) {
    throw new Error(`Invalid axis ${axis}, axis should be an integer.`);
  } else {
    if (axis < 0 || axis >= rank) {
      throw new Error(`Invalid axis ${axis}, axis should be in the interval [0, ${rank}).`);
    }
  }
  const inputShape = inputs[0].shape;
  const outputShape = inputShape.slice();
  for (let i = 1; i < inputs.length; ++i) {
    if (inputs[i].rank !== rank) {
      throw new Error('All input tensors should have the same rank.');
    } else {
      const shape = inputs[i].shape;
      for (let j = 0; j < inputShape.length; ++j) {
        if (j !== axis) {
          if (inputShape[j] !== shape[j]) {
            throw new Error('All input tensors should have the same shape, ' +
              'except for the size of the dimension to concatenate on.');
          }
        } else {
          outputShape[axis] += shape[axis];
        }
      }
    }
  }
  const output = new Tensor(outputShape);
  for (let i = 0; i < sizeOfShape(outputShape); ++i) {
    const location = output.locationFromIndex(i);
    let dim = location[axis];
    let k = 0;
    // Find out input k and its dim of axis according to output dim of axis
    for (; k < inputs.length; ++k) {
      if (dim < inputs[k].shape[axis]) {
        break;
      }
      dim -= inputs[k].shape[axis];
    }
    location[axis] = dim;
    const inputValue = inputs[k].getValueByLocation(location);
    output.setValueByIndex(i, inputValue);
  }
  return output;
}
