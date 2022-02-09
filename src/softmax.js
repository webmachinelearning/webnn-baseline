'use strict';

import {div, sub} from './binary.js';
import {exp} from './unary.js';
import {reduceMax, reduceSum} from './reduce.js';
import {validateSoftmaxParams} from './lib/validate-input.js';

/**
 * Compute the softmax values of the 2-D input tensor along axis 1.
 * @param {Tensor} x
 * @return {Tensor}
 */
export function softmax(x) {
  validateSoftmaxParams(...arguments);
  const maxX = reduceMax(x, {axes: [1], keepDimensions: true});
  const expX = exp(sub(x, maxX));
  return div(expX, reduceSum(expX, {axes: [1], keepDimensions: true}));
}
