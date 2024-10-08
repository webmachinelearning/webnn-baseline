'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateGatherParams} from './lib/validate-input.js';

/**
 * Gather values of the input tensor along an axis according to the indices.
 * @param {Tensor} input
 * @param {Tensor} indices
 * @param {MLGatherOptions} [options]
 * @return {Tensor}
 */
export function gather(input, indices, {axis = 0} = {}) {
  validateGatherParams(...arguments);
  const inputShape = input.shape;
  const outputShape = inputShape.slice(0, axis).concat(indices.shape, inputShape.slice(axis + 1));
  const output = new Tensor(outputShape);

  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    // output[i, j, k, ...] = input[indices[i, j, k, ...], j, k, ...] // if axis == 0
    // output[i, j, k, ...] = input[i, indices[i, j, k, ...], k, ...] // if axis == 1
    // output[i, j, k, ...] = input[i, j, indices[i, j, k, ...], ...] // if axis == 2
    const outputLocation = output.locationFromIndex(outputIndex);
    const indicesLocation = outputLocation.slice(axis, axis + indices.rank);
    let indiceValue = indices.getValueByLocation(indicesLocation);
    indiceValue = indiceValue < 0 ? indiceValue + input.shape[axis] : indiceValue;
    const selectedInputLocation = [
      ...outputLocation.slice(0, axis), indiceValue, ...outputLocation.slice(axis + indices.rank)];
    const inputValue = input.getValueByLocation(selectedInputLocation);
    output.setValueByIndex(outputIndex, inputValue);
  }

  return output;
}
