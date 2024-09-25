'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateCumulativeSumParams} from './lib/validate-input.js';

/**
 * Computes the cumulative sum of the input tensor along the specified axis.
 * @param {Tensor} input
 * @param {number} axis
 * @param {MLCumulativeSumOptions} options
 * @return {Tensor}
 */
export function cumulativeSum(input, axis, {exclusive = 0, reverse = 0} = {}) {
  validateCumulativeSumParams(...arguments);
  const inputShape = input.shape;
  const outputShape = [...inputShape];
  const output = new Tensor(outputShape);
  const numElementsAlongAxis = inputShape[axis];

  const totalElements = sizeOfShape(outputShape);

  for (let outputIndex = 0; outputIndex < totalElements; outputIndex++) {
    const loc = output.locationFromIndex(outputIndex);
    let cumulativeSumValue = 0;

    const start = reverse ? numElementsAlongAxis - 1 : 0;
    const step = reverse ? -1 : 1;
    const end = reverse ? -1 : numElementsAlongAxis;

    for (let i = start; reverse ? i > end : i < end; i += step) {
      const inputLoc = [...loc];
      inputLoc[axis] = exclusive ? (reverse ? i + 1 : i - 1) : i;

      if (!exclusive || (exclusive && inputLoc[axis] >= 0 &&
        inputLoc[axis] < numElementsAlongAxis)) {
        cumulativeSumValue += input.getValueByLocation(inputLoc);
      }

      const outputLoc = [...loc];
      outputLoc[axis] = i;
      output.setValueByLocation(outputLoc, cumulativeSumValue);
    }
  }

  return output;
}
