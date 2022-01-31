'use strict';

import {Tensor} from './lib/tensor.js';

/**
 * Clamp the input tensor element-wise within a range specified by the minimum and maximum values.
 * @param {Tensor} input
 * @param {MLClampOptions} [options]
 * @return {Tensor}
 */
export function clamp(input, options = {}) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < input.size; ++i) {
    const x = input.getValueByIndex(i);
    let y;
    if (options.minValue === undefined) {
      if (options.maxValue === undefined) {
        y = x;
      } else {
        y = Math.min(x, options.maxValue);
      }
    } else {
      if (options.maxValue === undefined) {
        y = Math.max(x, options.minValue);
      } else {
        y = Math.min(Math.max(x, options.minValue), options.maxValue);
      }
    }
    output.setValueByIndex(i, y);
  }
  return output;
}
