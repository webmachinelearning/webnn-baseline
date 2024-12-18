'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateGatherNDParams} from './lib/validate-input.js';

/**
 * Gathers values using multidimensional indices.
 * @param {Tensor} input
 * @param {Tensor} indices
 * @return {Tensor}
 */
export function gatherND(input, indices) {
  validateGatherNDParams(input, indices);

  const inputRank = input.rank;
  const inputShape = input.shape;
  const indicesRank = indices.rank;
  const indicesShape = indices.shape;
  const lastIndicesSize = indicesShape[indicesRank - 1];
  const tmpShape = inputShape.slice(lastIndicesSize);

  // Refer to https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/ +
  //   operation-specs/movement/gather-nd-8.html
  let outputShape = indicesShape.slice(0, indicesRank - 1);
  if (lastIndicesSize !== inputRank) {
    outputShape = outputShape.concat(tmpShape);
  }

  const output = new Tensor(outputShape);
  const indicesTotal = sizeOfShape(indicesShape);
  const tmp = new Tensor(tmpShape);
  const tmpTotal = sizeOfShape(tmpShape);

  for (let indicesIndex = 0; indicesIndex < indicesTotal; indicesIndex += lastIndicesSize) {
    const indicesLocation = indices.locationFromIndex(indicesIndex);
    const indicesArray = [];
    for (let i = 0; i < lastIndicesSize; i++) {
      const indicesValue = indices.getValueByIndex(indicesIndex + i);
      indicesArray.push(indicesValue >= 0 ? indicesValue : inputShape[i] + indicesValue);
    }
    for (let tmpIndex = 0; tmpIndex < tmpTotal; ++tmpIndex) {
      const tmpLocation = tmp.locationFromIndex(tmpIndex);
      const outputLocation = indicesLocation.slice(0, indicesRank - 1).concat(tmpLocation);
      const inputValue = input.getValueByLocation(indicesArray.concat(tmpLocation));
      // output[i_0, ..., i_{K-2},:,...,:] = data[indices[i_0, ..., i_{K-2}],:,...,:]
      output.setValueByLocation(outputLocation, inputValue);
    }
  }

  return output;
}
