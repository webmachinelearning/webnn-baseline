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
export function cumulativeSum(input, axis, {exclusive = false, reverse = false} = {}) {
  validateCumulativeSumParams(...arguments);
  const inputShape = input.shape;
  const outputShape = inputShape;
  const output = new Tensor(outputShape);
  const elementCountAlongAxis = inputShape[axis];
  const totalElements = sizeOfShape(outputShape);
  const inputElementStart = reverse ? elementCountAlongAxis - 1 : 0;
  const inputElementStep = reverse ? -1 : 1;

  const cumulativeSums = new Array(elementCountAlongAxis).fill(0);

  for (let outputIndex = 0; outputIndex < totalElements; outputIndex++) {
    const location = output.locationFromIndex(outputIndex);

    if (location[axis] !== inputElementStart) continue;

    for (let i = 0; i < elementCountAlongAxis; ++i) {
      const index = inputElementStart + i * inputElementStep;
      location[axis] = index;
      const inputValue = input.getValueByLocation(location);
      cumulativeSums[i] = (i === 0 ? 0 : cumulativeSums[i - 1]) + inputValue;
    }

    for (let i = 0; i < elementCountAlongAxis; ++i) {
      const index = inputElementStart + i * inputElementStep;
      location[axis] = index;
      const outputValue = exclusive ? (i === 0 ? 0 : cumulativeSums[i - 1]) : cumulativeSums[i];
      output.setValueByLocation(location, outputValue);
    }
  }

  return output;
}
