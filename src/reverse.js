'use strict';

import {sizeOfShape, Tensor} from './lib/tensor.js';
import {validateAxes} from './lib/validate-input.js';

/**
 * Reverse input along the given axes.
 * @param {Tensor} input
 * @param {MLReverseOptions} [options]
 * @return {Tensor}
 */
export function reverse(input, {axes}) {
  validateAxes(input, {axes});

  const inputAxes = axes ?? new Array(input.rank).fill(0).map((_, i) => i);
  const inputShape = input.shape;
  const outputShape = inputShape.slice();
  const outputSize = sizeOfShape(outputShape);
  const output = new Tensor(outputShape);

  for (let outputIndex = 0; outputIndex < outputSize; ++outputIndex) {
    const outputLocation = output.locationFromIndex(outputIndex);
    const selectedInputLocation = outputLocation.slice();
    for (const axis of inputAxes) {
      const index = selectedInputLocation[axis];
      selectedInputLocation[axis] = inputShape[axis] - index - 1;
    }
    const selectedInputValue = input.getValueByLocation(selectedInputLocation);
    output.setValueByLocation(outputLocation, selectedInputValue);
  }

  return output;
}
