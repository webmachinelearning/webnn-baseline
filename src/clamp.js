'use strict';

import {Tensor} from './lib/tensor.js';

/**
 * Clamp the input tensor element-wise within a range specified by the minimum and maximum values.
 * @param {Tensor} input
 * @param {MLClampOptions} [options]
 * @return {Tensor}
 */
export function clamp(input, {minValue=-Infinity, maxValue=Infinity} = {}) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < input.size; ++i) {
    const x = input.getValueByIndex(i);
    const y = Math.min(Math.max(x, minValue), maxValue);
    output.setValueByIndex(i, y);
  }
  return output;
}
