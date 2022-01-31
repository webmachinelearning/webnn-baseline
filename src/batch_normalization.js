'use strict';

import {add, sub, mul, div, pow} from './binary.js';
import {reshape} from './lib/reshape.js';
import {Tensor, Scalar} from './lib/tensor.js';

/**
 * Check the tensor whether it is a 1-D tensor and its length is equal to `expectedSize`.
 * @param {Tensor} a
 * @param {Number} expectedSize
 * @param {String} name
 */
function check1DTensorWithSize(a, expectedSize, name) {
  if (a) {
    if (a.rank !== 1) {
      throw new Error(`The parameter ${name} is not a 1-D tensor.`);
    } else {
      if (a.shape[0] !== expectedSize) {
        throw new Error(`The length ${a.shape[0]} of the ${name} values is not equal to the ` +
          `size ${expectedSize} of the input dimension denoted by options.axis.`);
      }
    }
  }
}

/**
 * Normalize the tensor values of input features across the batch dimension using
 * [Batch-Normalization](http://arxiv.org/abs/1502.03167).
 * @param {Tensor} input
 * @param {Tensor} mean
 * @param {Tensor} variance
 * @param {MLBatchNormalizationOptions} [options]
 * @return {Tensor}
 */
export function batchNormalization(input, mean, variance, options = {}) {
  let axis = options.axis;
  if (axis !== undefined) {
    if (!Number.isInteger(axis)) {
      throw new Error(`Invalid axis ${axis}, axis should be an integer.`);
    }
  } else {
    axis = 1;
  }
  const dim = input.shape[axis];
  check1DTensorWithSize(mean, dim, 'mean');
  check1DTensorWithSize(variance, dim, 'variance');
  const scale = options.scale;
  check1DTensorWithSize(scale, dim, 'scale');
  const bias = options.bias;
  check1DTensorWithSize(bias, dim, 'bias');
  const epsilon = options.epsilon ? options.epsilon : 1e-5;
  const activation = options.activation;
  // The output tensor of the same shape as the input tensor.
  let output = new Tensor(input.shape);
  const shape = new Array(input.rank).fill(1);
  shape[axis] = -1;
  output = sub(input, reshape(mean, shape));
  output = div(output,
      pow(add(reshape(variance, shape), new Scalar(epsilon)), new Scalar(0.5)));
  if (scale) {
    output = mul(output, reshape(scale, shape));
  }
  if (bias) {
    output = add(output, reshape(bias, shape));
  }
  if (activation) {
    output = activation(output);
  }
  return output;
}
