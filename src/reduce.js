'use strict';

import {squeeze} from './squeeze.js';
import {sizeOfShape, Tensor} from './tensor.js';

/**
 * Reduce the input along the dimensions given in axes.
 * @param {Tensor} input
 * @param {Function} reduceFunc
 * @param {MLReduceOptions} options
 * @return {Tensor}
 */
function reduce(input, reduceFunc, options = {}) {
  const axes = options.axes ? options.axes :
      new Array(input.rank).fill(0).map((_, i) => i);
  const keepDimensions = options.keepDimensions ? options.keepDimensions : false;

  if (axes.length > input.rank) {
    throw new Error(`The length ${axes.length} of axes is bigger than input rank ${input.rank}.`);
  }

  const outputShape = input.shape.slice();
  for (let i = 0; i < axes.length; ++i) {
    if (axes[i] === -1) {
      axes[i] = input.rank - 1;
    }
    if (axes[i] < 0 || axes[i] >= input.rank) {
      throw new Error(`The value ${axes[i]} at axis ${i} of axes is invalid.`);
    }
    outputShape[axes[i]] = 1;
  }

  // Calculate the "strides" across the reduction dimensions given in axes.
  axes.sort((a, b) => a - b);
  const reduceDims = axes.map((axis) => input.shape[axis]);
  const reduceElements = sizeOfShape(reduceDims);
  const reduceStrides = new Array(axes.length);
  reduceStrides[reduceStrides.length - 1] = 1;
  for (let i = reduceStrides.length - 2; i >= 0; --i) {
    reduceStrides[i] = reduceStrides[i + 1] * reduceDims[i + 1];
  }

  let output = new Tensor(outputShape);
  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const valuesToReduce = [];
    // Find all values to reduce.
    for (let reduceIndex = 0; reduceIndex < reduceElements; ++reduceIndex) {
      // Calculate the input location given index of elements to reduce.
      const inputLocation = output.locationFromIndex(outputIndex);
      let remainingReduceIndex = reduceIndex;
      for (let i = 0; i < axes.length; ++i) {
        const axis = axes[i];
        inputLocation[axis] = Math.floor(remainingReduceIndex / reduceStrides[i]);
        remainingReduceIndex -= inputLocation[axis] * reduceStrides[i];
      }
      valuesToReduce.push(input.getValueByLocation(inputLocation));
    }
    const outputValue = valuesToReduce.reduce(reduceFunc);
    output.setValueByIndex(outputIndex, outputValue);
  }

  if (!keepDimensions) {
    output = squeeze(output);
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
