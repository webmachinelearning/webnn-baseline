'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateScatterNDParams} from './lib/validate-input.js';
import {identity} from './unary.js';

/**
 * Scatter values using multidimensional indices.
 * @param {Tensor} input
 * @param {Tensor} indices
 * @param {Tensor} updates
 * @return {Tensor}
 */
export function scatterND(input, indices, updates) {
  // Refer to https://onnx.ai/onnx/operators/onnx__ScatterND.html
  // output = np.copy(data)
  // update_indices = indices.shape[:-1]
  // for idx in np.ndindex(update_indices):
  //     output[indices[idx]] = updates[idx]

  validateScatterNDParams(input, indices, updates);

  const output = identity(input);
  const inputRank = input.rank;
  const inputShape = input.shape;
  const indicesRank = indices.rank;
  const indicesShape = indices.shape;
  const indicesSize = sizeOfShape(indicesShape);
  const lastIndicesSize = indicesShape[indicesRank - 1];
  const sliceShape = inputShape.slice(lastIndicesSize, inputRank);
  const slice = new Tensor(sliceShape);
  const sliceSize = sizeOfShape(sliceShape);

  for (let indicesIndex = 0; indicesIndex < indicesSize; indicesIndex += lastIndicesSize) {
    const indicesLocation = indices.locationFromIndex(indicesIndex);
    const indicesArray = [];
    for (let i = 0; i < lastIndicesSize; i++) {
      const indicesValue = indices.getValueByIndex(indicesIndex + i);
      indicesArray.push(indicesValue >= 0 ? indicesValue : inputShape[i] + indicesValue);
    }
    for (let sliceIndex = 0; sliceIndex < sliceSize; ++sliceIndex) {
      const sliceLocation = slice.locationFromIndex(sliceIndex);
      const outputLocation = indicesArray.concat(sliceLocation);
      const updateValue =
        updates.getValueByLocation(indicesLocation.slice(0, indicesRank - 1).concat(sliceLocation));
      output.setValueByLocation(outputLocation, updateValue);
    }
  }

  return output;
}
