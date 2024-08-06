'use strict';

import {Tensor} from './lib/tensor.js';
import {validateConvTranspose2dParams} from './lib/validate-input.js';
import {transpose} from './transpose.js';

/**
 * Compute a 2-D transposed convolution given 4-D input and filter tensors.
 * @param {Tensor} input
 * @param {Tensor} filter
 * @param {MLConvTranspose2dOptions} options
 * @return {Tensor}
 */
export function convTranspose2d(
    input,
    filter,
    {
      padding = [0, 0, 0, 0],
      strides = [1, 1],
      groups = 1,
      dilations = [1, 1],
      outputPadding = [0, 0],
      outputSizes,
      activation = (x) => x,
      inputLayout = 'nchw',
      filterLayout = 'iohw',
      bias,
    } = {}) {
  // Below codes are using conv2d logic to compute convTranspose2d
  if (inputLayout === 'nhwc') {
    // nhwc -> nchw
    input = transpose(input, {permutation: [0, 3, 1, 2]});
  }
  if (filterLayout === 'iohw') {
    // iohw -> oihw, oihw is the default filterLayout of conv2d
    filter = transpose(filter, {permutation: [1, 0, 2, 3]});
  } else if (filterLayout === 'hwoi') {
    // hwoi -> oihw
    filter = transpose(filter, {permutation: [2, 3, 0, 1]});
  } else if (filterLayout === 'ohwi') {
    // ohwi -> oihw
    filter = transpose(filter, {permutation: [0, 3, 1, 2]});
  }

  validateConvTranspose2dParams(input, filter, {groups, bias});

  const [batchCount, inputChannels, inputHeight, inputWidth] = input.shape;
  const [outputChannelsPerGroup, , filterHeight, filterWidth] = filter.shape;
  const [strideHeight, strideWidth] = strides;
  const [dilationHeight, dilationWidth] = dilations;
  const effectiveFilterHeight = (filterHeight - 1) * dilationHeight + 1;
  const effectiveFilterWidth = (filterWidth - 1) * dilationWidth + 1;

  const [beginningPaddingHeight, endingPaddingHeight, beginningPaddingWidth, endingPaddingWidth] =
 padding;

  const outputShape = new Array(4);
  let outputHeight;
  let outputWidth;
  outputShape[0] = batchCount;

  const outputChannels = outputChannelsPerGroup * groups;
  outputShape[1] = outputChannels;

  if (outputSizes === undefined) {
    // output size = (input size - 1) * stride + filter size + (filter size - 1) * (dilation - 1) -
    //               beginning padding - ending padding + output padding
    outputHeight =
      (inputHeight - 1) * strideHeight + effectiveFilterHeight - beginningPaddingHeight -
       endingPaddingHeight + outputPadding[0];
    outputWidth =
      (inputWidth - 1) * strideWidth + effectiveFilterWidth - beginningPaddingWidth -
      endingPaddingWidth + outputPadding[1];
  } else {
    outputHeight = outputSizes[0];
    outputWidth = outputSizes[1];
  }

  outputShape[2] = outputHeight;
  outputShape[3] = outputWidth;
  let output = new Tensor(outputShape);

  // real padding = dilation * (kernel_size - 1) - padding, referring to
  //   https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
  const realBeginningPaddingHeight =
    dilationHeight * (filter.shape[2] - 1) - beginningPaddingHeight;
  const realBeginningPaddingWidth =
    dilationWidth * (filter.shape[3] - 1) - beginningPaddingWidth;
  const [realStrideHeight, realStrideWidth] = [1, 1];

  const inputChannelsPerGroup = inputChannels / groups;

  for (let ib = 0; ib < batchCount; ++ib) {
    for (let g = 0; g < groups; ++g) {
      for (let oc = 0; oc < outputChannelsPerGroup; ++oc) {
        for (let ic = 0; ic < inputChannelsPerGroup; ++ic) {
          for (let ih = -realBeginningPaddingHeight, oh = 0; oh < outputHeight;
            ih += realStrideHeight, ++oh) {
            for (let iw = -realBeginningPaddingWidth, ow = 0; ow < outputWidth;
              iw += realStrideWidth, ++ow) {
              const effectiveOutputChannel = oc + g * outputChannelsPerGroup;
              const outputLocation = [ib, effectiveOutputChannel, oh, ow];
              // make filter rotate 180
              for (let kh = filterHeight + (filterHeight - 1) * (dilationHeight - 1) - 1;
                kh >= 0; --kh) {
                for (let kw = filterWidth + (filterWidth - 1) * (dilationWidth - 1) - 1;
                  kw >= 0; --kw) {
                  const realKh = kh / dilationHeight;
                  const realKw = kw / dilationWidth;
                  const realIh = Math.floor((ih + kh)/ strideHeight);
                  const realIw = Math.floor((iw + kw)/ strideWidth);
                  if (realIh < 0 || realIh >= inputHeight ||
                    realIw < 0 || realIw >= inputWidth ||
                    (strideHeight > 1 && (ih + kh) % strideHeight !== 0) ||
                    (strideWidth > 1 && (iw + kw) % strideWidth !== 0) ||
                    (dilationHeight > 1 && kh % dilationHeight !==0) ||
                    (dilationWidth > 1 && kw % dilationWidth !==0)) {
                    // Skip the padding values.
                    continue;
                  } else {
                    const effectiveInputChannel = ic + g * inputChannelsPerGroup;
                    const inputValue = input.getValueByLocation(
                        [ib, effectiveInputChannel, realIh, realIw]);
                    // make filter rotate 180
                    const filterValue = filter.getValueByLocation(
                        [oc, ic, filterHeight - realKh - 1,
                          filterWidth - realKw - 1]);
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
