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
    outputShape[dimensionIndex] = Math.floor(sizes[dimensionIndex] / strides[dimensionIndex]) +
      Number(sizes[dimensionIndex] % strides[dimensionIndex]);
  }

  const output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const loc = output.locationFromIndex(outputIndex);
    const selectedInputLoc = loc.slice();
    for (let i = 0; i < loc.length; ++i) {
      selectedInputLoc[i] = starts[i] + loc[i] * strides[i];
    }
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }
  return output;
}
