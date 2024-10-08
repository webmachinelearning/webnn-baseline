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

  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const outputLocation = output.locationFromIndex(outputIndex);
    const indicesLocation = outputLocation.slice();
    let indiceVlaue = indices.getValueByLocation(indicesLocation);
    indiceVlaue = indiceVlaue < 0 ? indiceVlaue + input.shape[axis] : indiceVlaue;
    const selectedInputLoc =
        [...outputLocation.slice(0, axis), indiceVlaue, ...outputLocation.slice(axis + 1)];
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }

  return output;
}
