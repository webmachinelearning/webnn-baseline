'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateGatherElementsParams} from './lib/validate-input.js';

/**
 * GatherElements takes elements from the input tensor at positions specified in the indices tensor.
 * @param {Tensor} input
 * @param {Tensor} indices
 * @param {MLGatherOptions} [options]
 * @return {Tensor}
 */
export function gatherElements(input, indices, {axis = 0} = {}) {
  validateGatherElementsParams(...arguments);
  const outputShape = indices.shape.slice();
  const output = new Tensor(outputShape);
  const count = sizeOfShape(indices.shape);

  for (let outputIndex = 0; outputIndex < count; ++outputIndex) {
    const outputLocation = output.locationFromIndex(outputIndex);
    const indicesLocation = outputLocation.slice();
    let indicesValue = indices.getValueByLocation(indicesLocation);
    indicesValue = indicesValue < 0 ? indicesValue + input.shape[axis] : indicesValue;
    const selectedInputLocation =
        [...outputLocation.slice(0, axis), indicesValue, ...outputLocation.slice(axis + 1)];
    const inputValue = input.getValueByLocation(selectedInputLocation);
    output.setValueByIndex(outputIndex, inputValue);
  }

  return output;
}
