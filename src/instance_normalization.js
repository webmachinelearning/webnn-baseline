'use strict';

import {add, sub, mul, div, pow} from './binary.js';
import {reduceMean} from './reduce.js';
import {reshape} from './reshape.js';
import {transpose} from './transpose.js';
import {Tensor, Scalar} from './lib/tensor.js';
import {validateInstanceNormalizationParams} from './lib/validate-input.js';

/**
 * Normalize the input features using [Instance-Normalization](https://arxiv.org/abs/1607.08022).
 * @param {Tensor} input
 * @param {MLInstanceNormalizationOptions} [options]
 * @return {Tensor}
 */
export function instanceNormalization(
    input,
    {
      scale,
      bias,
      epsilon=1e-5,
      layout='nchw',
    } = {}) {
  validateInstanceNormalizationParams(...arguments);
  if (layout === 'nhwc') {
    // nhwc -> nchw
    input = transpose(input, {permutation: [0, 3, 1, 2]});
  }
  // The output tensor has the same shape as the input tensor.
  let output = new Tensor(input.shape);
  const shape = [1, null, 1, 1];
  const reduceOptions = {axes: [2, 3], keepDimensions: true};
  const mean = reduceMean(input, reduceOptions);
  output = sub(input, reshape(mean, shape));
  const variance = reduceMean(pow(output, new Scalar(2)), reduceOptions);
  output = div(output,
      pow(add(reshape(variance, shape), new Scalar(epsilon)), new Scalar(0.5)));
  if (scale) {
    output = mul(output, reshape(scale, shape));
  }
  if (bias) {
    output = add(output, reshape(bias, shape));
  }
  if (layout === 'nhwc') {
    // nchw -> nhwc
    output = transpose(output, {permutation: [0, 2, 3, 1]});
  }
  return output;
}
