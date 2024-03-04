'use strict';

import {pow} from './binary.js';
import {squeeze} from './reshape.js';
import {abs, exp, log} from './unary.js';
import {sizeOfShape, Scalar, Tensor} from './lib/tensor.js';
import {validateReduceParams} from './lib/validate-input.js';

export function selectValuesToReduce(input, axes, inputLocation) {
  validateReduceParams(input, {axes});

  const outputShape = input.shape.slice();
  for (let i = 0; i < axes.length; ++i) {
    outputShape[axes[i]] = 1;
  }

  // Calculate the "strides" across the reduction dimensions given in axes.
  axes.sort((a, b) => a - b);
  const reduceDims = axes.map((axis) => input.shape[axis]);
  const reducedElementCount = sizeOfShape(reduceDims);
  const reduceStrides = new Array(axes.length);

  if (reduceStrides.length > 0) {
    reduceStrides[reduceStrides.length - 1] = 1;
  }

  if (reduceStrides.length > 1) {
    for (let i = reduceStrides.length - 2; i >= 0; --i) {
      reduceStrides[i] = reduceStrides[i + 1] * reduceDims[i + 1];
    }
  }

  const valuesToReduce = [];
  // Find all values to reduce.
  for (let reduceIndex = 0; reduceIndex < reducedElementCount; ++reduceIndex) {
    // Calculate the input location given index of elements to reduce.
    let remainingReduceIndex = reduceIndex;
    for (let i = 0; i < axes.length; ++i) {
      const axis = axes[i];
      inputLocation[axis] = Math.floor(remainingReduceIndex / reduceStrides[i]);
      remainingReduceIndex -= inputLocation[axis] * reduceStrides[i];
    }
    valuesToReduce.push(input.getValueByLocation(inputLocation));
  }

  return valuesToReduce;
}

/**
 * Reduce the input along the dimensions given in axes.
 * @param {Tensor} input
 * @param {Function} reduceFunc
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
function reduce(input, reduceFunc, {keepDimensions = false, axes} = {}) {
  const inputAxes = axes ?? new Array(input.rank).fill(0).map((_, i) => i);

  if (inputAxes.length === 0) {
    return input;
  }

  const outputShape = input.shape.slice();
  for (let i = 0; i < inputAxes.length; ++i) {
    outputShape[inputAxes[i]] = 1;
  }

  let output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const inputLocation = output.locationFromIndex(outputIndex);
    const valuesToReduce = selectValuesToReduce(input, inputAxes, inputLocation);
    const outputValue = valuesToReduce.reduce(reduceFunc);
    output.setValueByIndex(outputIndex, outputValue);
  }

  if (!keepDimensions) {
    output = squeeze(output, {axes});
  }
  return output;
}

/* The max reducer */
export const maxReducer = (previousValue, currentValue) => Math.max(previousValue, currentValue);

/**
 * Compute the maximum value of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceMax(input, options = {}) {
  return reduce(input, maxReducer, options);
}

/* The mean reducer */
export function meanReducer(previousValue, currentValue, currentIndex, array) {
  if (currentIndex === array.length - 1) {
    return (previousValue + currentValue) / array.length;
  } else {
    return previousValue + currentValue;
  }
}

/**
 * Compute the average value of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceMean(input, options = {}) {
  return reduce(input, meanReducer, options);
}

/**
 * Compute the minimum value of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceMin(input, options = {}) {
  return reduce(input,
      (previousValue, currentValue) => Math.min(previousValue, currentValue), options);
}

/**
 * Compute the product of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceProduct(input, options = {}) {
  return reduce(input,
      (previousValue, currentValue) => previousValue * currentValue, options);
}

/**
 * Compute the sum of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceSum(input, options = {}) {
  return reduce(input,
      (previousValue, currentValue) => previousValue + currentValue, options);
}

/**
 * Compute the sum of the square of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceSumSquare(input, options = {}) {
  return reduceSum(pow(input, new Scalar(2)), options);
}

/**
 * Compute the L1 norm of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceL1(input, options = {}) {
  return reduceSum(abs(input), options);
}

/**
 * Compute the L2 norm of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceL2(input, options = {}) {
  const intermediateResult = reduceSumSquare(input, options);
  if (intermediateResult.rank === 0) {
    return new Tensor(
        [],
        [Math.pow(intermediateResult.getValueByIndex(0), 0.5)]);
  } else {
    return pow(intermediateResult, new Scalar(0.5));
  }
}

/**
 * Compute the log value of the sum of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceLogSum(input, options = {}) {
  return log(reduceSum(input, options));
}

/**
 * Compute the log value of the sum of the exponent of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
export function reduceLogSumExp(input, options = {}) {
  return log(reduceSum(exp(input), options));
}
