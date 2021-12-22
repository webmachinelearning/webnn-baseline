'use strict';

import {Tensor} from './tensor.js';

/**
 * Permute the dimensions of the input tensor according to the permutation argument.
 * @param {Tensor} input
 * @param {MLTransposeOptions} [options]
 * @return {Tensor}
 */
export function transpose(input, options = {}) {
  const permutation = options.permutation ? options.permutation :
    new Array(input.rank).fill(0).map((e, i, a) => a.length - i - 1);
  if (permutation.length !== input.rank) {
    throw new Error('The permutation is invalid.');
  }

  const outputShape = new Array(input.rank).fill(0).map((e, i, a) => input.shape[permutation[i]]);
  const output = new Tensor(outputShape);
  for (let inputIndex = 0; inputIndex < input.data.length; ++inputIndex) {
    const inputValue = input.data[inputIndex];
    const inputLocation = input.locationFromIndex(inputIndex);
    const outputLocation = new Array(output.rank);
    for (let i = 0; i < permutation.length; ++i) {
      outputLocation[i] = inputLocation[permutation[i]];
    }
    output.setValue(outputLocation, inputValue);
  }
  return output;
}
