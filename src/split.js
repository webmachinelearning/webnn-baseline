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
  const rank = input.rank;
  const inpAxis = axis >=0 ? axis : rank + axis;
  if (typeof splits === 'number') {
    sliceSizes = new Array(splits).fill(input.shape[inpAxis] / splits);
  } else if (splits instanceof Array) {
    sliceSizes = splits.slice();
  }
  let start = 0;
  for (const size of sliceSizes) {
    outputs.push(slice(input, [start], [size], {axes: [inpAxis]}));
    start += size;
  }
  return outputs;
}
