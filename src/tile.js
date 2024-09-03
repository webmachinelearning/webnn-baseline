'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateTileParams} from './lib/validate-input.js';

/**
 * Represents the tile operation that repeats a tensor the given number of times along each axis.
 * @param {Tensor} input
 * @param {Array} repetitions
 * @return {Tensor}
 */
export function tile(input, repetitions) {
  validateTileParams(...arguments);
  const outputShape = input.shape.map((size, index) => {
    return size * repetitions[index];
  });
  const output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const loc = output.locationFromIndex(outputIndex);
    const selectedInputLoc = loc.slice();
    for (let i = 0; i < loc.length; ++i) {
      selectedInputLoc[i] = loc[i] % input.shape[i];
    }
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }
  return output;
}
