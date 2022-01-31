'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateInput} from './lib/validate-input.js';

/**
 * Concatenates the input tensors along a given axis.
 * @param {Array.<Tensor>} inputs
 * @param {Number} axis
 * @return {Tensor}
 */
export function concat(inputs, axis) {
  validateInput('concat', arguments);
  const inputShape = inputs[0].shape;
  const outputShape = inputShape.slice();
  for (let i = 1; i < inputs.length; ++i) {
    outputShape[axis] += inputs[i].shape[axis];
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
