'use strict';

import {cast} from './cast.js';
import {Tensor, sizeOfShape} from './lib/tensor.js';
import {reduceMax, reduceMin, selectValuesToReduce} from './reduce.js';
import {squeeze} from './squeeze.js';


/**
 * Get the index location of the minimum or maxmium values of all the input values along the axes.
 * @param {Tensor} input
 * @param {Number} axis
 * @param {Function} reduceFunc
 * @param {MLArgMinMaxOptions} [options]
 * @return {Tensor}
 */
export function argMaxMin(
    input,
    axis,
    reduceFunc,
    {
      keepDimensions = false,
      outputDatatype = 'int32',
    } = {}) {
  // If axes aren't present (defaulting to null), all dimensions are reduced.
  // See https://webmachinelearning.github.io/webnn/#dom-mlargminmaxoptions-axes.
  const axes=[axis];
  const inputAxes = axes ?? new Array(input.rank).fill(0).map((_, i) => i);
  const outputShape = input.shape.slice();

  for (let i = 0; i < inputAxes.length; ++i) {
    outputShape[inputAxes[i]] = 1;
  }

  let output = new Tensor(outputShape);
  const tensor = reduceFunc(input, {axes: inputAxes, keepDimensions: true});

  for (let outputIndex = 0; outputIndex < sizeOfShape(outputShape); ++outputIndex) {
    const value = tensor.getValueByIndex(outputIndex);
    const inputLocation = output.locationFromIndex(outputIndex);
    const selectedArray = selectValuesToReduce(input, inputAxes, inputLocation);
    const index =selectedArray.indexOf(value);
    output.setValueByIndex(outputIndex, index);
  }

  if (!keepDimensions) {
    output = squeeze(output, {axes});
  }

  return cast(output, outputDatatype);
}

/**
 * Get the index location of the maxmium values of all the input values along the axes.
 * @param {Tensor} input
 * @param {Number} axis
 * @param {MLArgMinMaxOptions} [options]
 * @return {Tensor}
 */
export function argMax(input, axis, options = {}) {
  return argMaxMin(input, axis, reduceMax, options);
}

/**
 * Get the index location of the minimum values of all the input values along the axes.
 * @param {Tensor} input
 * @param {Number} axis
 * @param {MLArgMinMaxOptions} [options]
 * @return {Tensor}
 */
export function argMin(input, axis, options = {}) {
  return argMaxMin(input, axis, reduceMin, options);
}
