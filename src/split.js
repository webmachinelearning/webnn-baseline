'use strict';

import {slice} from './slice.js';

/**
 * Split the input tensor into a number of sub tensors along the given axis.
 * @param {Tensor} input
 * @param {Array|Number} splits
 * @param {MLSplitOptions} options
 * @return {Array.<Tensor>}
 */
export function split(input, splits, options = {}) {
  const outputs = [];
  let sliceSizes = [];
  const rank = input.rank;
  let axis = 0;
  if (options.axis !== undefined) {
    if (!Number.isInteger(options.axis)) {
      throw new Error(`The axis ${options.axis} should be an integer.`);
    }
    if (options.axis >= rank || options.axis < -rank) {
      throw new Error(`The axis ${options.axis} should be in the interval [${-rank}, ${rank}).`);
    }
    axis = options.axis >= 0 ? options.axis : rank + options.axis;
  }
  if (typeof splits === 'number') {
    if (!Number.isInteger(splits) || splits <= 0) {
      throw new Error(`Invalid splits ${splits}, it should be a positive integer.`);
    }
    if (input.shape[axis] % splits !== 0) {
      throw new Error(`The splits ${splits} must evenly divide the dimension size ` +
        `${input.shape[axis]} of input along options.axis ${options.axis}.`);
    }
    sliceSizes = new Array(splits).fill(input.shape[axis] / splits);
  } else if (splits instanceof Array) {
    if (!splits.every((v) => Number.isInteger(v) && v > 0)) {
      throw new Error(`Invalid splits ${splits}, it should be an Array of positive integers.`);
    }
    const sum = splits.reduce((a, b) => a + b);
    if (sum !== input.shape[axis]) {
      throw new Error(`Invalid [${splits}], the sum of sizes ${sum} must equal to the dimension ` +
        `size ${input.shape[axis]} of input along options.axis ${options.axis}`);
    }
    sliceSizes = splits.slice();
  }
  let start = 0;
  for (const size of sliceSizes) {
    outputs.push(slice(input, [start], [size], {axes: [axis]}));
    start += size;
  }
  return outputs;
}
