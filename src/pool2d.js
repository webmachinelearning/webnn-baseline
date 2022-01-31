'use strict';

import {computePaddingForAutoPad} from './lib/compute-padding.js';
import {Tensor} from './lib/tensor.js';
import {transpose} from './transpose.js';
import {meanReducer, maxReducer} from './reduce.js';

/**
 * Compute a reduction operation across all the elements within the
 * moving window over the input tensor.
 * @param {Tensor} input
 * @param {Function} reductionFunc
 * @param {MLPool2dOptions} options
 * @return {Tensor}
 */
function pool2d(input, reductionFunc, options = {}) {
  if (input.rank !== 4) {
    throw new Error('The input should be a 4-D tensor.');
  }

  const padding = options.padding ? options.padding : [0, 0, 0, 0];
  const strides = options.strides ? options.strides : [1, 1];
  const dilations = options.dilations ? options.dilations : [1, 1];
  const roundingType = options.roundingType ? options.roundingType : 'floor';
  if (roundingType !== 'floor' && roundingType !== 'ceil') {
    throw new Error('The rounding type is invalid.');
  }
  const roundingFunc = roundingType === 'floor' ? Math.floor : Math.ceil;

  const layout = options.layout ? options.layout : 'nchw';
  if (layout === 'nhwc') {
    // nhwc -> nchw
    input = transpose(input, {permutation: [0, 3, 1, 2]});
  }

  const batchCount = input.shape[0];
  const channels = input.shape[1];
  const inputHeight = input.shape[2];
  const inputWidth = input.shape[3];
  const windowDimensions = options.windowDimensions ?
    options.windowDimensions : [inputHeight, inputWidth];
  const windowHeight = windowDimensions[0];
  const windowWidth = windowDimensions[1];
  const strideHeight = strides[0];
  const strideWidth = strides[1];
  const dilationHeight = dilations[0];
  const dilationWidth = dilations[1];
  const effectiveWindowHeight = windowHeight + (windowHeight - 1) * (dilationHeight - 1);
  const effectiveWindowWidth = windowWidth + (windowWidth - 1) * (dilationWidth - 1);

  let beginningPaddingHeight;
  let endingPaddingHeight;
  let beginningPaddingWidth;
  let endingPaddingWidth;
  if (options.autoPad === undefined || options.autoPad === 'explicit') {
    [beginningPaddingHeight, endingPaddingHeight, beginningPaddingWidth, endingPaddingWidth] =
      padding;
  } else {
    [beginningPaddingHeight, endingPaddingHeight] = computePaddingForAutoPad(
        options.autoPad, inputHeight, effectiveWindowHeight, strideHeight);
    [beginningPaddingWidth, endingPaddingWidth] = computePaddingForAutoPad(
        options.autoPad, inputWidth, effectiveWindowWidth, strideWidth);
  }

  const outputShape = new Array(4);
  outputShape[0] = batchCount;
  outputShape[1] = channels;
  const outputHeight = options.outputSizes ? options.outputSizes[0] :
    roundingFunc(
        1 + (inputHeight - effectiveWindowHeight + beginningPaddingHeight + endingPaddingHeight) /
          strideHeight);
  outputShape[2] = outputHeight;
  const outputWidth = options.outputSizes ? options.outputSizes[1] :
    roundingFunc(
        1 + (inputWidth - effectiveWindowWidth + beginningPaddingWidth + endingPaddingWidth) /
          strideWidth);
  outputShape[3] = outputWidth;
  let output = new Tensor(outputShape);

  for (let ib = 0; ib < batchCount; ++ib) {
    for (let ic = 0; ic < channels; ++ic) {
      for (let ih = -beginningPaddingHeight, oh = 0; oh < outputHeight; ih += strideHeight, ++oh) {
        for (let iw = -beginningPaddingWidth, ow = 0; ow < outputWidth; iw += strideWidth, ++ow) {
          const outputLocation = [ib, ic, oh, ow];
          const valuesInWindow = [];
          for (let kh = 0; kh < windowHeight; ++kh) {
            for (let kw = 0; kw < windowWidth; ++kw) {
              const dkh = kh * dilationHeight;
              const dkw = kw * dilationWidth;
              if (ih + dkh < 0 || ih + dkh >= inputHeight ||
                  iw + dkw < 0 || iw + dkw >= inputWidth) {
                // Skip the padding values.
                continue;
              } else {
                const inputValue = input.getValueByLocation(
                    [ib, ic, ih + dkh, iw + dkw]);
                valuesInWindow.push(inputValue);
              }
            }
          }
          const outputValue = valuesInWindow.reduce(reductionFunc);
          output.setValueByLocation(outputLocation, outputValue);
        }
      }
    }
  }

  if (layout === 'nhwc') {
    // nchw -> nhwc
    output = transpose(output, {permutation: [0, 2, 3, 1]});
  }

  return output;
}

/**
 * Compute a mean reduction operation across all the elements within the moving window over
 * the input tensor.
 * @param {Tensor} input
 * @param {MLPool2dOptions} options
 * @return {Tensor}
 */
export function averagePool2d(input, options = {}) {
  return pool2d(input, meanReducer, options);
}

/**
 * Compute a max reduction operation across all the elements within the moving window over
 * the input tensor.
 * @param {Tensor} input
 * @param {MLPool2dOptions} options
 * @return {Tensor}
 */
export function maxPool2d(input, options = {}) {
  return pool2d(input, maxReducer, options);
}
