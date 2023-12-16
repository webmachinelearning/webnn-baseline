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
  const shapeInput = input.shape;

  // set outputShape following Spec Algorithm
  //   https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-gather
  //
  // let dimCount = 0
  // let rankOutput = 0;
  // let shapeOutput = [];
  // for (dimCount = 0; dimCount < shapeInput.length; dimCount++) {
  //   if (dimCount === axis) {
  //     break;
  //   } else {
  //     shapeOutput[dimCount] = shapeInput[dimCount];
  //   }
  // }
  // rankOutput = dimCount;
  // for (dimCount = 0; dimCount < indices.shape.length; dimCount++) {
  //   shapeOutput[rankOutput + dimCount] = indices.shape[dimCount];
  // }
  // rankOutput = rankOutput + dimCount;
  // for (dimCount = 0; dimCount < shapeInput.length; dimCount++) {
  //   if (dimCount <= axis) {
  //     continue;
  //   } else {
  //     shapeOutput[rankOutput + dimCount - axis - 1] = shapeInput[dimCount];
  //   }
  // }

  // optimized set outputShape using JavaScript slice and concat
  const shapeOutput = shapeInput.slice(0, axis).concat(indices.shape, shapeInput.slice(axis + 1));
  const output = new Tensor(shapeOutput);

  for (let outputIndex = 0; outputIndex < sizeOfShape(shapeOutput); ++outputIndex) {
    // output[i, j, k, ...] = input[indices[i, j, k, ...], j, k, ...] // if axis == 0
    // output[i, j, k, ...] = input[i, indices[i, j, k, ...], k, ...] // if axis == 1
    // output[i, j, k, ...] = input[i, j, indices[i, j, k, ...], ...] // if axis == 2
    const outputLoc = output.locationFromIndex(outputIndex);
    const indicesLoc = outputLoc.slice(axis, axis + indices.rank);
    const selectedInputLoc = outputLoc.slice(0, axis)
        .concat(indices.getValueByLocation(indicesLoc), outputLoc.slice(axis + indices.rank));
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }

  return output;
}
