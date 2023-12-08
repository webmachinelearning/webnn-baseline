'use strict';

import {add, sub, mul, div, pow} from './binary.js';
import {reduceMean} from './reduce.js';
import {reshape} from './reshape.js';
import {sqrt} from './unary.js';
import {Tensor, Scalar} from './lib/tensor.js';
import {transpose} from './transpose.js';
import {validateLayerNormalizationParams} from './lib/validate-input.js';

/**
 * Sort the indexes of the elements in the axes array
 * based on their values and return the sorted index array
 * @param {Array} axes
 * @return {Array}
 */
export function getIndexOfSortedValue(axes) {
  const sortedIndices = axes.map((_, index) => index);
  sortedIndices.sort((a, b) => axes[a] - axes[b]);
  return sortedIndices;
}

/**
 * Normalize the tensor values of input features using
 * [layer-Normalization](https://arxiv.org/abs/1607.06450)
 * @param {Tensor} input
 * @param {MLLayerNormalizationOptions} [options]
 * @return {Tensor}
 */
export function layerNormalization(input, {scale, bias, axes, epsilon=1e-5}) {
  validateLayerNormalizationParams(...arguments);
  if (axes === undefined) {
    axes = Array.from({length: input.rank - 1}, (_, index) => index + 1);
  }
  const sortAxes = getIndexOfSortedValue(axes);
  if (scale) {
    scale = transpose(scale, {permutation: sortAxes});
  }
  if (bias) {
    bias = transpose(bias, {permutation: sortAxes});
  }
  // The output tensor has the same shape as the input tensor.
  let output = new Tensor(input.shape);
  const inputShape = input.shape;
  const compatibleShape = new Array(input.rank).fill(1);
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    compatibleShape[axis] = inputShape[axis];
  }
  const reduceOptions = {axes, keepDimensions: true};
  const mean = reduceMean(input, reduceOptions);
  const variance = reduceMean(pow(sub(input, mean), new Scalar(2)), reduceOptions);
  output = div(sub(input, mean), sqrt(add(variance, new Scalar(epsilon))));
  if (scale) {
    output = mul(output, reshape(scale, compatibleShape));
  }
  if (bias) {
    output = add(output, reshape(bias, compatibleShape));
  }
  return output;
}
