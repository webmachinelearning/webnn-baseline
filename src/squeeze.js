'use strict';

import {reshape} from './reshape.js';
import {validateInput} from './lib/validate-input.js';

/**
 * Reduce the rank of a tensor by eliminating dimensions with size 1 of the tensor shape.
 * @param {Tensor} input
 * @param {MLSqueezeOptions} options
 * @return {Tensor}
 */
export function squeeze(input, {axes} = {}) {
  validateInput('squeeze', arguments);
  const inpAxes = axes ?? new Array(input.rank).fill(0).map((_, i) => i);

  const outputShape = input.shape.filter((dim, axis) =>
    !(dim === 1 && inpAxes.indexOf(axis) !== -1));
  const output = reshape(input, outputShape);
  return output;
}
