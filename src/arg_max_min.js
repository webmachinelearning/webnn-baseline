'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {reduceMax, reduceMin, selectValuesToReduce} from './reduce.js';
import {squeeze} from './squeeze.js';

/**
 * Get the index location of the minimum or maxmium values of all the input values along the axes.
 * @param {Tensor} input
 * @param {Function} reduceFunc
 * @param {MLArgMinMaxOptions} [options]
 * @return {Tensor}
 */
export function argMaxMin(
    input,
    reduceFunc,
    {
      axes = null,
      keepDimensions = false,
      selectLastIndex = false,
    } = {}) {
  // If axes doesn't present (defaulting to null), all dimensions are reduced.
  // See https://webmachinelearning.github.io/webnn/#dom-mlargminmaxoptions-axes.
  const inpAxes = axes ?? new Array(input.rank).fill(0).map((_, i) => i);
  const outputShape = input.shape.slice();

  for (let i = 0; i < inpAxes.length; ++i) {
    outputShape[inpAxes[i]] = 1;
  }

  let output = new Tensor(outputShape);
  const tensor = reduceFunc(input, {axes: inpAxes, keepDimensions: true});

  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const value = tensor.getValueByIndex(outputIndex);
    const inputLocation = output.locationFromIndex(outputIndex);
    const selectedArray = selectValuesToReduce(input, inpAxes, inputLocation);
    const index =
        selectLastIndex ? selectedArray.lastIndexOf(value) : selectedArray.indexOf(value);
    output.setValueByIndex(outputIndex, index);
  }

  if (!keepDimensions) {
    output = squeeze(output, {axes});
  }

  return output;
}

/**
 * Get the index location of the maxmium values of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLArgMinMaxOptions} [options]
 * @return {Tensor}
 */
export function argMax(input, options = {}) {
  return argMaxMin(input, reduceMax, options);
}

/**
 * Get the index location of the minimum values of all the input values along the axes.
 * @param {Tensor} input
 * @param {MLArgMinMaxOptions} [options]
 * @return {Tensor}
 */
export function argMin(input, options = {}) {
  return argMaxMin(input, reduceMin, options);
}
