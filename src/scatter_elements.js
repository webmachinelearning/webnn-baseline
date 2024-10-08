'use strict';

import {sizeOfShape} from './lib/tensor.js';
import {validateScatterElementsParams} from './lib/validate-input.js';
import {identity} from './unary.js';

/**
 * Create a copy of the input data, and then update its value to values specified by updates at
 * specific index positions specified by indices.
 * @param {Tensor} input
 * @param {Tensor} indices
 * @param {Tensor} updates
 * @param {MLScatterOptions} [options]
 * @return {Tensor}
 */
export function scatterElements(input, indices, updates, {axis = 0} = {}) {
  validateScatterElementsParams(...arguments);
  const output = identity(input);

  for (let indicesIndex = 0; indicesIndex < sizeOfShape(indices.shape); ++indicesIndex) {
    // output[indices[i, j, k, ...], j, k, ...] = updates[i, j, k, ...] // if axis == 0
    // output[i, indices[i, j, k, ...], k, ...] = updates[i, j, k, ...] // if axis == 1
    // output[i, j, indices[i, j, k, ...], ...] = updates[i, j, k, ...] // if axis == 2
    const indicesLocation = indices.locationFromIndex(indicesIndex);
    let indicesValue = indices.getValueByIndex(indicesIndex);
    indicesValue = indicesValue < 0 ? indicesValue + input.shape[axis] : indicesValue;
    const outputLocation =
        [...indicesLocation.slice(0, axis), indicesValue, ...indicesLocation .slice(axis + 1)];
    output.setValueByLocation(outputLocation, updates.getValueByIndex(indicesIndex));
  }

  return output;
}
