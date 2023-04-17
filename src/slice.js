'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateSliceParams} from './lib/validate-input.js';

/**
 * Produce a slice of the input tensor.
 * @param {Tensor} input
 * @param {Array} starts
 * @param {Array} sizes
 * @return {Tensor}
 */
export function slice(input, starts, sizes) {
  validateSliceParams(...arguments);
  const outputShape = sizes;
  const output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const loc = output.locationFromIndex(outputIndex);
    const selectedInputLoc = loc.slice();
    for (let i = 0; i < loc.length; ++i) {
      selectedInputLoc[i] = loc[i] + starts[i];
    }
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }
  return output;
}
