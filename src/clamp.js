'use strict';

import {Tensor} from './tensor.js';

/**
 * Clamp the input tensor element-wise within a range specified by the minimum and maximum values.
 * @param {Tensor} input
 * @param {MLClampOptions} [options]
 * @return {Tensor}
 */
export function clamp(input, options = {}) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < output.data.length; ++i) {
    const x = input.data[i];
    if (options.minValue === undefined) {
      if (options.maxValue === undefined) {
        output.data[i] = x;
      } else {
        output.data[i] = Math.min(x, options.maxValue);
      }
    } else {
      if (options.maxValue === undefined) {
        output.data[i] = Math.max(x, options.minValue);
      } else {
        output.data[i] = Math.min(Math.max(x, options.minValue), options.maxValue);
      }
    }
  }
  return output;
}
