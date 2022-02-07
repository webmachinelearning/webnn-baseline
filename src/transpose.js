'use strict';

import {Tensor} from './lib/tensor.js';
import {validateInput} from './lib/validate-input.js';

/**
 * Permute the dimensions of the input tensor according to the permutation argument.
 * @param {Tensor} input
 * @param {MLTransposeOptions} [options]
 * @return {Tensor}
 */
export function transpose(input, {permutation} = {}) {
  const inpPermutation = permutation ??
        new Array(input.rank).fill(0).map((e, i, a) => a.length - i - 1);
  validateInput('transpose', [input, {permutation: inpPermutation}]);

  const outputShape = new Array(input.rank).fill(0).map(
      (e, i, a) => input.shape[inpPermutation[i]]);
  const output = new Tensor(outputShape);
  for (let inputIndex = 0; inputIndex < input.size; ++inputIndex) {
    const inputValue = input.getValueByIndex(inputIndex);
    const inputLocation = input.locationFromIndex(inputIndex);
    const outputLocation = new Array(output.rank);
    for (let i = 0; i < inpPermutation.length; ++i) {
      outputLocation[i] = inputLocation[inpPermutation[i]];
    }
    output.setValueByLocation(outputLocation, inputValue);
  }
  return output;
}
