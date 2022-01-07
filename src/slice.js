'use strict';

import {Tensor, sizeOfShape} from './tensor.js';

/**
 * Update the location with a given offset.
 * @param {Array} x
 * @param {Array} offset
 * @return {Array}
 */
function updateLocation(x, offset) {
  const out = x.slice();
  for (let i = 0; i < x.length; ++i) {
    out[i] = x[i] + offset[i];
  }
  return out;
}

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
  const allStarts = new Array(rank).fill(0);
  let axes = options.axes;
  if (axes) {
    if (axes.length > rank) {
      throw new Error('The length of axes is invalid.');
    } else {
      for (let i = 0; i < axes.length; ++i) {
        if (!Number.isInteger(axes[i]) || axes[i] >= rank || axes[i] < -rank) {
          throw new Error('The value of axes is invalid.');
        }
      }
    }
  } else {
    axes = [...Array(rank).keys()];
  }
  const axesLen = axes.length;
  if (starts.length !== axesLen) {
    throw new Error('The length of starts is invalid.');
  }
  if (sizes.length !== axesLen) {
    throw new Error('The length of sizes is invalid.');
  }
  const outputShape = input.shape.slice();
  for (let j = 0; j < axesLen; ++j) {
    const slicedDim = axes[j] >= 0 ? axes[j] : axes[j] + rank;
    const slicedDimSize = input.shape[slicedDim];
    const slicedStart = starts[j];
    allStarts[slicedDim] = slicedStart >= 0 ? slicedStart : slicedStart + slicedDimSize;
    if (!Number.isInteger(slicedStart) || slicedStart >= slicedDimSize ||
        slicedStart < -slicedDimSize) {
      throw new Error('The value of starts is invalid.');
    } else {
      const slicedSize = sizes[j];
      if (Number.isInteger(slicedSize) && slicedSize >= 0) {
        if (slicedStart >= 0) {
          if (slicedStart + slicedSize > slicedDimSize) {
            throw new Error('The value of sizes is invalid.');
          } else {
            outputShape[slicedDim] = slicedSize;
          }
        } else {
          if (slicedStart + slicedSize > 0) {
            throw new Error('The value of sizes is invalid.');
          } else {
            outputShape[slicedDim] = slicedSize;
          }
        }
      } else {
        if (slicedSize !== -1) {
          throw new Error('The value of sizes is invalid.');
        } else {
          outputShape[slicedDim] = slicedStart >= 0 ? slicedDimSize - slicedStart : -slicedStart;
        }
      }
    }
  }
  const output = new Tensor(outputShape);
  for (let k = 0; k < sizeOfShape(outputShape); ++k) {
    const loc = output.locationFromIndex(k);
    const selectedInputLoc = updateLocation(loc, allStarts);
    const inputValue = input.getValueByLocation(selectedInputLoc);
    output.setValueByIndex(k, inputValue);
  }
  return output;
}
