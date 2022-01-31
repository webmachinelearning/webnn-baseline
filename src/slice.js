'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';

/**
 * Produce a slice of the input tensor.
 * @param {Tensor} input
 * @param {Array} starts
 * @param {Array} sizes
 * @param {MLSliceOptions} options
 * @return {Tensor}
 */
export function slice(input, starts, sizes, options = {}) {
  const rank = input.rank;
  const startsForAllAxes = new Array(rank).fill(0);
  let axes = options.axes;
  if (axes) {
    if (axes.length > rank) {
      throw new Error(`The length of axes ${axes.length} is greater than rank ${rank}.`);
    } else {
      for (const axis of axes) {
        if (!Number.isInteger(axis)) {
          throw new Error(`Invalid axes value ${axis}, it should be an integer.`);
        } else {
          if (axis >= rank || axis < -rank) {
            throw new Error(`Invalid axes value ${axis}, it should be in the interval ` +
              `[${-rank}, ${rank}).`);
          }
        }
      }
    }
  } else {
    axes = [...Array(rank).keys()];
  }
  const axesLen = axes.length;
  if (starts.length !== axesLen) {
    throw new Error(`The length ${starts.length} of starts is not equal to the length ` +
      `${axesLen} of axes.`);
  }
  if (sizes.length !== axesLen) {
    throw new Error(`The length ${sizes.length} of sizes is not equal to the length ${axesLen} ` +
      'of axes.');
  }
  const outputShape = input.shape.slice();
  for (let i = 0; i < axesLen; ++i) {
    const axis = axes[i] >= 0 ? axes[i] : axes[i] + rank;
    const size = input.shape[axis];
    const start = starts[i];
    if (!Number.isInteger(start)) {
      throw new Error(`Invalid starts value ${start}, it should be an integer.`);
    }
    startsForAllAxes[axis] = start >= 0 ? start : start + size;
    if (start >= size || start < -size) {
      throw new Error(`Invalid starts value ${start}, it shoule be in the interval ` +
        `[${-size}, ${size}).`);
    } else {
      const sliceSize = sizes[i];
      if (!Number.isInteger(sliceSize)) {
        throw new Error(`Invalid sizes value ${sliceSize}, it should be an integer.`);
      }
      if (sliceSize >= 0) {
        if (start >= 0) {
          if (start + sliceSize > size) {
            throw new Error(`Invalid sizes value ${sliceSize}, the sum of the start ${start} ` +
              `plus the size ${sliceSize} is greater than the dimensional size ${size}`);
          } else {
            outputShape[axis] = sliceSize;
          }
        } else {
          if (start + sliceSize > 0) {
            throw new Error(`Invalid sizes value ${sliceSize}, the sum of the start ${start} ` +
              `plus the size ${sliceSize} is greater than the dimensional size ${size}`);
          } else {
            outputShape[axis] = sliceSize;
          }
        }
      } else {
        if (sliceSize !== -1) {
          throw new Error(`The value ${sliceSize} of sizes is invalid, it is required to be -1 ` +
            'when it is negative.');
        } else {
          outputShape[axis] = start >= 0 ? size - start : -start;
        }
      }
    }
  }
  const output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const loc = output.locationFromIndex(outputIndex);
    const selectedInputLoc = loc.slice();
    for (let i = 0; i < loc.length; ++i) {
      selectedInputLoc[i] = loc[i] + startsForAllAxes[i];
    }
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(outputIndex, inputValue);
  }
  return output;
}
