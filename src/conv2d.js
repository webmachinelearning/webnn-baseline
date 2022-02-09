'use strict';

import {Tensor} from './lib/tensor.js';
import {validateConv2dParams} from './lib/validate-input.js';
import {computePaddingForAutoPad} from './lib/compute-padding.js';
import {transpose} from './transpose.js';

/**
 * Compute a 2-D convolution given 4-D input and filter tensors.
 * @param {Tensor} input
 * @param {Tensor} filter
 * @param {MLConv2dOptions} options
 * @return {Tensor}
 */
export function conv2d(input, filter, {padding = [0, 0, 0, 0],
  strides = [1, 1],
  groups = 1,
  dilations = [1, 1],
  activation = (x) => x,
  inputLayout = 'nchw',
  filterLayout = 'oihw',
  bias,
  autoPad = 'explicit',
}
= {}) {
  if (inputLayout === 'nhwc') {
    // nhwc -> nchw
    input = transpose(input, {permutation: [0, 3, 1, 2]});
  }
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
  validateConv2dParams(input, filter, {groups, bias});

  const [batchCount, inputChannels, inputHeight, inputWidth] = input.shape;
  const [outputChannels, , filterHeight, filterWidth] = filter.shape;
  const [strideHeight, strideWidth] = strides;
  const [dilationHeight, dilationWidth] = dilations;
  const effectiveFilterHeight = filterHeight + (filterHeight - 1) * (dilationHeight - 1);
  const effectiveFilterWidth = filterWidth + (filterWidth - 1) * (dilationWidth - 1);

  let beginningPaddingHeight;
  let endingPaddingHeight;
  let beginningPaddingWidth;
  let endingPaddingWidth;
  if (autoPad === 'explicit') {
    [beginningPaddingHeight, endingPaddingHeight, beginningPaddingWidth, endingPaddingWidth] =
      padding;
  } else {
    [beginningPaddingHeight, endingPaddingHeight] = computePaddingForAutoPad(
        autoPad, inputHeight, effectiveFilterHeight, strideHeight);
    [beginningPaddingWidth, endingPaddingWidth] = computePaddingForAutoPad(
        autoPad, inputWidth, effectiveFilterWidth, strideWidth);
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

  output = activation(output);

  if (inputLayout === 'nhwc') {
    // nchw -> nhwc
    output = transpose(output, {permutation: [0, 2, 3, 1]});
  }

  return output;
}
