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
  const shapeOutput = indices.shape.slice();
  const output = new Tensor(shapeOutput);

  for (let outputIndex = 0; outputIndex < sizeOfShape(shapeOutput); ++outputIndex) {
    const outputLoc = output.locationFromIndex(outputIndex);
    const indicesLoc = outputLoc.slice();
    let indiceVlaue = indices.getValueByLocation(indicesLoc);
    indiceVlaue = indiceVlaue < 0 ? indiceVlaue + input.shape[axis] : indiceVlaue;
    const selectedInputLoc =
        [...outputLoc.slice(0, axis), indiceVlaue, ...outputLoc.slice(axis + 1)];
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }

  return output;
}
