'use strict';

import {Tensor} from './tensor.js';
import {transpose} from './transpose.js';

// TODO: implement autoPad
// TODO: implement dilations
// TODO: implement groups
// TODO: implement activation
// TODO: implement bias

/**
 * Compute a 2-D convolution given 4-D input and filter tensors.
 * @param {Tensor} input
 * @param {Tensor} filter
 * @param {MLConv2dOptions} options
 * @return {Tensor}
 */
export function conv2d(input, filter, options = {}) {
  if (input.rank !== 4) {
    throw Error('The input should be a 4-D tensor.');
  }

  if (filter.rank !== 4) {
    throw Error('The filter should be a 4-D tensor.');
  }

  const padding = options.padding ? options.padding : [0, 0, 0, 0];
  const strides = options.strides ? options.strides : [1, 1];
  const groups = options.groups ? options.groups : 1;
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
  const beginningPaddingHeight = padding[0];
  const endingPaddingHeight = padding[1];
  const beginningPaddingWidth = padding[2];
  const endingPaddingWidth = padding[3];
  const strideHeight = strides[0];
  const strideWidth = strides[1];

  if (inputChannels !== filterInputChannels * groups) {
    throw Error('The input channels of filter is invalid.');
  }

  const outputShape = new Array(4);
  outputShape[0] = batchCount;
  outputShape[1] = outputChannels;
  const outputHeight = 1 + (inputHeight - filterHeight +
    beginningPaddingHeight + endingPaddingHeight) / strideHeight;
  outputShape[2] = outputHeight;
  const outputWidth = 1 + (inputWidth - filterWidth +
    beginningPaddingWidth + endingPaddingWidth) / strideWidth;
  outputShape[3] = outputWidth;

  let output = new Tensor(outputShape);
  for (let ib = 0; ib < batchCount; ++ib) {
    for (let oc = 0; oc < outputChannels; ++oc) {
      for (let ic = 0; ic < inputChannels; ++ic) {
        for (let ih = -beginningPaddingHeight, oh = 0;
          ih + filterHeight <= inputHeight + endingPaddingHeight;
          ih += strideHeight, ++oh) {
          for (let iw = -beginningPaddingWidth, ow = 0;
            iw + filterWidth <= inputWidth + endingPaddingWidth;
            iw += strideWidth, ++ow) {
            const outputLocation = [ib, oc, oh, ow];
            for (let kh = 0; kh < filterHeight; ++kh) {
              for (let kw = 0; kw < filterWidth; ++kw) {
                let inputValue;
                if (ih + kh < 0 || ih + kh >= inputHeight || iw + kw < 0 || iw + kw >= inputWidth) {
                  // Zero padding.
                  inputValue = 0;
                } else {
                  inputValue = input.getValue([ib, ic, ih + kh, iw + kw]);
                }
                const filterValue = filter.getValue([oc, ic, kh, kw]);
                output.setValue(outputLocation,
                    output.getValue(outputLocation) + inputValue * filterValue);
              }
            }
          }
        }
      }
    }
  }

  if (inputLayout === 'nhwc') {
    // nchw -> nhwc
    output = transpose(output, {permutation: [0, 2, 3, 1]});
  }

  return output;
}
