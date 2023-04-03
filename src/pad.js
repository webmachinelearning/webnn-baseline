'use strict';

import {Tensor} from './lib/tensor.js';

/**
 * Get mapped location from source tensor.
 * @param {Array} location
 * @param {Array} inputShape
 * @param {Array} beginningPadding
 * @param {MLPaddingMode} mode
 * @return {Array} mapped location of source tensor
 */
function getMappedLocation(location, inputShape, beginningPadding, mode) {
  const mappedLocation = location.slice();
  const rank = location.length;
  if (mode === 'edge') {
    for (let i = 0; i < rank; i++) {
      if (location[i] < beginningPadding[i]) {
        mappedLocation[i] = 0;
      } else if (location[i] >= beginningPadding[i] + inputShape[i]) {
        mappedLocation[i] = inputShape[i] - 1;
      } else {
        mappedLocation[i] -= beginningPadding[i];
      }
    }
  } else {
    // reflection mode or symmetric mode
    const offset = mode === 'symmetric' ? 1 : 0;
    for (let i = 0; i < rank; i++) {
      if (mappedLocation[i] < beginningPadding[i]) {
        // mappedLocation[i] =
        //    beginningPadding[i] + (beginningPadding[i] - mappedLocation[i]) -
        //    beginningPadding[i] - offset;
        mappedLocation[i] = beginningPadding[i] - mappedLocation[i] - offset;
      } else if (mappedLocation[i] >= beginningPadding[i] + inputShape[i]) {
        // mappedLocation[i] =
        //    beginningPadding[i] + inputShape[i] - 1 -
        //    (mappedLocation[i] - (beginningPadding[i] + inputShape[i] -1)) -
        //    beginningPadding[i] + offset;
        mappedLocation[i] =
            beginningPadding[i] + inputShape[i] * 2 - mappedLocation[i] - 2 + offset;
      } else {
        mappedLocation[i] -= beginningPadding[i];
      }
    }
  }
  return mappedLocation;
}

/**
 * Update element value of destination tensor.
 * @param {Number} index
 * @param {Tensor} source
 * @param {Tensor} destination
 * @param {Array} beginningPadding
 * @param {MLPaddingMode} mode
 * @param {Number} value
 * @return {Tensor}
 */
function updateOutputElement(index, source, destination, beginningPadding, mode, value) {
  const sourceShape = source.shape;
  const location = destination.locationFromIndex(index);
  const rank = location.length;
  let hasFillPadding = false;
  for (let j = 0; j < rank; j++) {
    if (location[j] < beginningPadding[j] || location[j] >= beginningPadding[j] + sourceShape[j]) {
      hasFillPadding = true;
      break;
    }
  }
  let result;
  if (hasFillPadding) {
    if (mode === 'constant') {
      result = value;
    } else if (mode === 'edge' || mode === 'reflection' || mode === 'symmetric') {
      const targetLocation = getMappedLocation(location, sourceShape, beginningPadding, mode);
      result = source.getValueByLocation(targetLocation);
    }
  } else {
    const inputLocation = location.map((v, d) => v - beginningPadding[d]);
    result = source.getValueByLocation(inputLocation);
  }
  destination.setValueByIndex(index, result);
  return destination;
}

/**
 * Inflate the tensor with constant or mirrored values on the edges.
 * @param {Tensor} input
 * @param {Array} beginningPadding
 * @param {Array} endingPadding
 * @param {MLPadOptions} [options]
 * @return {Tensor}
 */
export function pad(
    input,
    beginningPadding,
    endingPadding,
    {
      mode='constant',
      value=0,
    } = {}) {
  const outputShape = input.shape.map((v, i) => v + beginningPadding[i] + endingPadding[i]);
  let output = new Tensor(outputShape);
  for (let i = 0; i < output.size; ++i) {
    output = updateOutputElement(i, input, output, beginningPadding, mode, value);
  }
  return output;
}
