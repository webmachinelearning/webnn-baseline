'use strict';

import {Tensor} from './lib/tensor.js';
import {transpose} from './transpose.js';

/**
 * Resample the tensor values from the source to the destination spatial dimensions according to
 * the scaling factors using Nearest Neighbor interpolation.
 * Refer to https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
 * @param {Tensor} src
 * @param {Tensor} dst
 * @param {Number} dstHeight
 * @param {Number} dstWidth
 * @param {Number} scaleHeight
 * @param {Number} scaleWidth
 * @return {Tensor}
 */
function nearestNeighbor(src, dst, dstHeight, dstWidth, scaleHeight, scaleWidth) {
  const [srcBatches, srcChannels, srcHeight, srcWidth] = src.shape;
  for (let ob = 0; ob < srcBatches; ob++) {
    for (let oc = 0; oc < srcChannels; oc++) {
      for (let oh = 0; oh < dstHeight; oh++) {
        let ih = (oh + 0.5) / scaleHeight - 0.5;
        ih = Math.max(Math.min(ih, srcHeight - 1), 0);
        for (let ow = 0; ow < dstWidth; ow++) {
          let iw = (ow + 0.5) / scaleWidth - 0.5;
          iw = Math.max(Math.min(iw, srcWidth - 1), 0);
          const outputValue = src.getValueByLocation(
              [ob, oc, Math.ceil(ih - 0.5), Math.ceil(iw - 0.5)]);
          dst.setValueByLocation([ob, oc, oh, ow], outputValue);
        }
      }
    }
  }
  return dst;
}

/**
 * Resample the tensor values from the source to the destination spatial dimensions according to
 * the scaling factors using Bilinear interpolation.
 * Refer to https://en.wikipedia.org/wiki/Bilinear_interpolation
 * @param {Tensor} src
 * @param {Tensor} dst
 * @param {Number} dstHeight
 * @param {Number} dstWidth
 * @param {Number} scaleHeight
 * @param {Number} scaleWidth
 * @return {Tensor}
 */
function linear(src, dst, dstHeight, dstWidth, scaleHeight, scaleWidth) {
  const [srcBatches, srcChannels, srcHeight, srcWidth] = src.shape;
  for (let ob = 0; ob < srcBatches; ob++) {
    for (let oc = 0; oc < srcChannels; oc++) {
      for (let oh = 0; oh < dstHeight; oh++) {
        let ih = (oh + 0.5) / scaleHeight - 0.5;
        ih = Math.max(Math.min(ih, srcHeight - 1), 0);
        const ihLower = Math.floor(ih);
        const ihUpper = Math.ceil(ih);
        const u = ih - ihLower;
        for (let ow = 0; ow < dstWidth; ow++) {
          let iw = (ow + 0.5) / scaleWidth - 0.5;
          iw = Math.max(Math.min(iw, srcWidth - 1), 0);
          const iwLower = Math.floor(iw);
          const iwUpper = Math.ceil(iw);
          const v = iw - iwLower;
          const outputValue =
              (1 - u) * (1 - v) * src.getValueByLocation([ob, oc, ihLower, iwLower]) +
              (1 - u) * v * src.getValueByLocation([ob, oc, ihLower, iwUpper]) +
              u * (1 - v) * src.getValueByLocation([ob, oc, ihUpper, iwLower]) +
              u * v * src.getValueByLocation([ob, oc, ihUpper, iwUpper]);
          dst.setValueByLocation([ob, oc, oh, ow], outputValue);
        }
      }
    }
  }
  return dst;
}

const interpolationFunctions = {
  'nearest-neighbor': nearestNeighbor,
  'linear': linear,
};

/**
 * Resample the tensor values from the source to the destination spatial dimensions according to
 * the scaling factors.
 * @param {Tensor} input
 * @param {MLResample2dOptions} options
 * @return {Tensor}
 */
export function resample2d(
    input,
    {
      mode = 'nearest-neighbor',
      scales = [1.0, 1.0],
      sizes,
      axes = [2, 3],
    } = {},
) {
  let targetHeight;
  let targetWidth;
  let [scaleHeight, scaleWidth] = scales;
  // permute input by nchw layout
  if (axes[0] == 0 && axes[1] == 1) {
    // hwnc -> nchw
    input = transpose(input, {permutation: [2, 3, 0, 1]});
  } else if (axes[0] == 1 && axes[1] == 2) {
    // nhwc -> nchw
    input = transpose(input, {permutation: [0, 3, 1, 2]});
  } else if (axes[0] == 2 && axes[1] == 3) {
    // do nothing nchw -> nchw
  }
  if (sizes !== undefined) {
    [targetHeight, targetWidth] = sizes;
    scaleHeight = targetHeight / input.shape[2];
    scaleWidth = targetWidth / input.shape[3];
  } else {
    targetHeight = Math.floor(input.shape[2] * scaleHeight);
    targetWidth = Math.floor(input.shape[3] * scaleWidth);
  }
  const outputShape = input.shape.slice();
  outputShape[2] = targetHeight;
  outputShape[3] = targetWidth;
  let output = new Tensor(outputShape);
  output = interpolationFunctions[mode](
      input, output, targetHeight, targetWidth, scaleHeight, scaleWidth);
  if (axes[0] == 0 && axes[1] == 1) {
    // nchw -> hwnc
    output = transpose(output, {permutation: [2, 3, 0, 1]});
  } else if (axes[0] == 1 && axes[1] == 2) {
    // nchw -> nhwc
    output = transpose(output, {permutation: [0, 2, 3, 1]});
  } else if (axes[0] == 2 && axes[1] == 3) {
    // do nothing nchw -> nchw
  }
  return output;
}
