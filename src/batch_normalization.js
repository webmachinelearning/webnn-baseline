'use strict';

import {add, sub, mul, div, pow} from './binary.js';
import {reshape} from './reshape.js';
import {Tensor, Scalar} from './lib/tensor.js';
import {validateBatchNormalizationParams} from './lib/validate-input.js';

/**
 * Normalize the tensor values of input features across the batch dimension using
 * [Batch-Normalization](http://arxiv.org/abs/1502.03167).
 * @param {Tensor} input
 * @param {Tensor} mean
 * @param {Tensor} variance
 * @param {MLBatchNormalizationOptions} [options]
 * @return {Tensor}
 */
export function batchNormalization(input, mean, variance, {axis=1, scale, bias, epsilon=1e-5,
  activation = (x) => x} = {}) {
  validateBatchNormalizationParams(...arguments);
  // The output tensor has the same shape as the input tensor.
  let output = new Tensor(input.shape);
  const shape = new Array(input.rank).fill(1);
  shape[axis] = null;
  output = sub(input, reshape(mean, shape));
  output = div(output,
      pow(add(reshape(variance, shape), new Scalar(epsilon)), new Scalar(0.5)));
  if (scale) {
    output = mul(output, reshape(scale, shape));
  }
  if (bias) {
    output = add(output, reshape(bias, shape));
  }
  output = activation(output);
  return output;
}
