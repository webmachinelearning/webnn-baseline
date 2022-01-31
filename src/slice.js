'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateInput} from './lib/validate-input.js';

/**
 * Produce a slice of the input tensor.
 * @param {Tensor} input
 * @param {Array} starts
 * @param {Array} sizes
 * @param {MLSliceOptions} options
 * @return {Tensor}
 */
export function slice(input, starts, sizes, {axes} = {}) {
  const rank = input.rank;
  const startsForAllAxes = new Array(rank).fill(0);
  validateInput("slice", arguments);

  axes = axes ?? [...Array(rank).keys()];
  const axesLen = axes.length;
  const outputShape = input.shape.slice();
  for (let i = 0; i < axesLen; ++i) {
    const axis = axes[i] >= 0 ? axes[i] : axes[i] + rank;
    const size = input.shape[axis];
    const start = starts[i];
    startsForAllAxes[axis] = start >= 0 ? start : start + size;
    const sliceSize = sizes[i];
    if (sliceSize >= 0) {
      outputShape[axis] = sliceSize;
    } else {
      outputShape[axis] = start >= 0 ? size - start : -start;
    }
  }
  const output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const loc = output.locationFromIndex(outputIndex);
    const selectedInputLoc = loc.slice();
    for (let i = 0; i < loc.length; ++i) {
      selectedInputLoc[i] = loc[i] + startsForAllAxes[i];
    }
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }
  return output;
}
