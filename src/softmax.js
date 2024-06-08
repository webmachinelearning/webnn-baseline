'use strict';

import {div, sub} from './binary.js';
import {exp} from './unary.js';
import {reduceMax, reduceSum} from './reduce.js';
import {validateSoftmaxParams} from './lib/validate-input.js';

/**
 * Compute the softmax values of the N-D input tensor along the given axis.
 * @param {Tensor} input
 * @param {Number} axis
 * @return {Tensor}
 */
export function softmax(input, axis) {
  validateSoftmaxParams(...arguments);
  const maxX = reduceMax(input, {axes: [axis], keepDimensions: true});
  const expX = exp(sub(input, maxX));
  return div(expX, reduceSum(expX, {axes: [axis], keepDimensions: true}));
}
