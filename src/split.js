'use strict';

import {slice} from './slice.js';
import {validateSplitParams} from './lib/validate-input.js';

/**
 * Split the input tensor into a number of sub tensors along the given axis.
 * @param {Tensor} input
 * @param {Array|Number} splits
 * @param {MLSplitOptions} options
 * @return {Array.<Tensor>}
 */
export function split(input, splits, {axis = 0} = {}) {
  validateSplitParams(...arguments);
  const outputs = [];
  let sliceSizes = [];
  if (typeof splits === 'number') {
    sliceSizes = new Array(splits).fill(input.shape[axis] / splits);
  } else if (splits instanceof Array) {
    sliceSizes = splits.slice();
  }
  const starts = new Array(input.rank).fill(0);
  const sizes = input.shape.slice();
  let start = 0;
  for (const size of sliceSizes) {
    starts[axis] = start;
    sizes[axis] = size;
    outputs.push(slice(input, starts, sizes));
    start += size;
  }
  return outputs;
}
