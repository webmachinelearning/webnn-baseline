'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateSliceParams} from './lib/validate-input.js';

/**
 * Produce a slice of the input tensor.
 * @param {Tensor} input
 * @param {Array} starts
 * @param {Array} sizes
 * @param {MLSliceOptions} options
 * @return {Tensor}
 */
export function slice(input, starts, sizes, {strides} = {}) {
  validateSliceParams(...arguments);
  strides = strides ?? new Array(input.rank).fill(1);
  const outputShape = input.shape.slice();

  for (let dimensionIndex = 0; dimensionIndex < input.rank; ++dimensionIndex) {
    outputShape[dimensionIndex] = Math.ceil(sizes[dimensionIndex] / strides[dimensionIndex]);
  }

  const output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const outputLocation = output.locationFromIndex(outputIndex);
    const selectedInputLocation = outputLocation.slice();
    for (let i = 0; i < outputLocation.length; ++i) {
      selectedInputLocation[i] = starts[i] + outputLocation[i] * strides[i];
    }
    const inputValue = input.getValueByLocation(selectedInputLocation);
    output.setValueByIndex(outputIndex, inputValue);
  }
  return output;
}
