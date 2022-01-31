'use strict';

import {Tensor} from './lib/tensor.js';
import {computePaddingForAutoPad} from './lib/compute-padding.js';
import {transpose} from './transpose.js';

/**
 * Compute a 2-D convolution given 4-D input and filter tensors.
 * @param {Tensor} input
 * @param {Tensor} filter
 * @param {MLConv2dOptions} options
 * @return {Tensor}
 */
export function conv2d(input, filter, options = {}) {
  if (input.rank !== 4) {
    throw new Error('The input should be a 4-D tensor.');
  }

  if (filter.rank !== 4) {
    throw new Error('The filter should be a 4-D tensor.');
  }

  const padding = options.padding ? options.padding : [0, 0, 0, 0];
  const strides = options.strides ? options.strides : [1, 1];
  const groups = options.groups ? options.groups : 1;
  const dilations = options.dilations ? options.dilations : [1, 1];
  const activation = options.activation;

  const inputLayout = options.inputLayout ? options.inputLayout : 'nchw';
  if (inputLayout === 'nhwc') {
    // nhwc -> nchw
    input = transpose(input, {permutation: [0, 3, 1, 2]});
  }
  const filterLayout = options.filterLayout ? options.filterLayout : 'oihw';
  if (filterLayout === 'hwio') {
    // hwio -> oihw
    filter = transpose(filter, {permutation: [3, 2, 0, 1]});
  } else if (filterLayout === 'ohwi') {
    // ohwi -> oihw
    filter = transpose(filter, {permutation: [0, 3, 1, 2]});
  } else if (filterLayout === 'ihwo') {
    // ihwo -> oihw
    filter = transpose(filter, {permutation: [3, 0, 1, 2]});
  }

  const batchCount = input.shape[0];
  const inputChannels = input.shape[1];
  const inputHeight = input.shape[2];
  const inputWidth = input.shape[3];
  const outputChannels = filter.shape[0];
  const filterInputChannels = filter.shape[1];
  const filterHeight = filter.shape[2];
  const filterWidth = filter.shape[3];
  const strideHeight = strides[0];
  const strideWidth = strides[1];
  const dilationHeight = dilations[0];
  const dilationWidth = dilations[1];
  const effectiveFilterHeight = filterHeight + (filterHeight - 1) * (dilationHeight - 1);
  const effectiveFilterWidth = filterWidth + (filterWidth - 1) * (dilationWidth - 1);

  if (inputChannels !== filterInputChannels * groups) {
    throw new Error('The input channels of filter is invalid.');
  }

  const bias = options.bias;
  if (bias && (bias.rank !== 1 || bias.shape[0] != outputChannels)) {
    throw new Error('the bias should be a 1-D tensor with the shape of [output_channels].');
  }

  let beginningPaddingHeight;
  let endingPaddingHeight;
  let beginningPaddingWidth;
  let endingPaddingWidth;
  if (options.autoPad === undefined || options.autoPad === 'explicit') {
    [beginningPaddingHeight, endingPaddingHeight, beginningPaddingWidth, endingPaddingWidth] =
      padding;
  } else {
    [beginningPaddingHeight, endingPaddingHeight] = computePaddingForAutoPad(
        options.autoPad, inputHeight, effectiveFilterHeight, strideHeight);
    [beginningPaddingWidth, endingPaddingWidth] = computePaddingForAutoPad(
        options.autoPad, inputWidth, effectiveFilterWidth, strideWidth);
  }

  const outputShape = new Array(4);
  outputShape[0] = batchCount;
  outputShape[1] = outputChannels;
  const outputHeight =
    1 + (inputHeight - effectiveFilterHeight + beginningPaddingHeight + endingPaddingHeight) /
      strideHeight;
  outputShape[2] = outputHeight;
  const outputWidth =
    1 + (inputWidth - effectiveFilterWidth + beginningPaddingWidth + endingPaddingWidth) /
      strideWidth;
  outputShape[3] = outputWidth;
  let output = new Tensor(outputShape);

  const outputChannelsPerGroup = outputChannels / groups;
  const inputChannelsPerGroup = inputChannels / groups;

  for (let ib = 0; ib < batchCount; ++ib) {
    for (let g = 0; g < groups; ++g) {
      for (let oc = 0; oc < outputChannelsPerGroup; ++oc) {
        for (let ic = 0; ic < inputChannelsPerGroup; ++ic) {
          for (let ih = -beginningPaddingHeight, oh = 0; oh < outputHeight;
            ih += strideHeight, ++oh) {
            for (let iw = -beginningPaddingWidth, ow = 0; ow < outputWidth;
              iw += strideWidth, ++ow) {
              const effectiveOutputChannel = oc + g * outputChannelsPerGroup;
              const outputLocation = [ib, effectiveOutputChannel, oh, ow];
              for (let kh = 0; kh < filterHeight; ++kh) {
                for (let kw = 0; kw < filterWidth; ++kw) {
                  const dkh = kh * dilationHeight;
                  const dkw = kw * dilationWidth;
                  if (ih + dkh < 0 || ih + dkh >= inputHeight ||
                      iw + dkw < 0 || iw + dkw >= inputWidth) {
                    // Skip the padding values.
                    continue;
                  } else {
                    const effectiveInputChannel = ic + g * inputChannelsPerGroup;
                    const inputValue = input.getValueByLocation(
                        [ib, effectiveInputChannel, ih + dkh, iw + dkw]);
                    const filterValue = filter.getValueByLocation(
                        [effectiveOutputChannel, ic, kh, kw]);
                    let outputValue = output.getValueByLocation(outputLocation);
                    outputValue += inputValue * filterValue;
                    output.setValueByLocation(outputLocation, outputValue);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (bias) {
    for (let ib = 0; ib < batchCount; ++ib) {
      for (let oc = 0; oc < outputChannels; ++oc) {
        for (let oh = 0; oh < outputHeight; ++oh) {
          for (let ow = 0; ow < outputWidth; ++ow) {
            const outputLocation = [ib, oc, oh, ow];
            const biasValue = bias.getValueByLocation([oc]);
            let outputValue = output.getValueByLocation(outputLocation);
            outputValue += biasValue;
            output.setValueByLocation(outputLocation, outputValue);
          }
        }
      }
    }
  }

  if (activation) {
    output = activation(output);
  }

  if (inputLayout === 'nhwc') {
    // nchw -> nhwc
    output = transpose(output, {permutation: [0, 2, 3, 1]});
  }

  return output;
}
