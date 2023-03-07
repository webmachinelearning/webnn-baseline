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
  const [srcBatches, srcHeight, srcWidth, srcChannels] = src.shape;
  for (let ob = 0; ob < srcBatches; ob++) {
    for (let oc = 0; oc < srcChannels; oc++) {
      for (let oh = 0; oh < dstHeight; oh++) {
        let ih = (oh + 0.5) / scaleHeight - 0.5;
        if (ih < 0) {
          ih = 0;
        } else if (ih > srcHeight - 1) {
          ih = srcHeight - 1;
        }
        for (let ow = 0; ow < dstWidth; ow++) {
          let iw = (ow + 0.5) / scaleWidth - 0.5;
          if (iw < 0) {
            iw = 0;
          } else if (iw > srcWidth - 1) {
            iw = srcWidth - 1;
          }
          const outputValue = src.getValueByLocation(
              [ob, Math.ceil(ih - 0.5), Math.ceil(iw - 0.5), oc]);
          dst.setValueByLocation([ob, oh, ow, oc], outputValue);
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
  const [srcBatches, srcHeight, srcWidth, srcChannels] = src.shape;
  for (let ob = 0; ob < srcBatches; ob++) {
    for (let oc = 0; oc < srcChannels; oc++) {
      for (let oh = 0; oh < dstHeight; oh++) {
        let ih = (oh + 0.5) / scaleHeight - 0.5;
        if (ih < 0) {
          ih = 0;
        } else if (ih > srcHeight - 1) {
          ih = srcHeight - 1;
        }
        const ihLower = Math.floor(ih);
        const ihUpper = Math.ceil(ih);
        const u = ih - ihLower;
        for (let ow = 0; ow < dstWidth; ow++) {
          let iw = (ow + 0.5) / scaleWidth - 0.5;
          if (iw < 0) {
            iw = 0;
          } else if (iw > srcWidth - 1) {
            iw = srcWidth - 1;
          }
          const iwLower = Math.floor(iw);
          const iwUpper = Math.ceil(iw);
          const v = iw - iwLower;
          const outputValue =
              (1 - u) * (1 - v) * src.getValueByLocation([ob, ihLower, iwLower, oc]) +
              (1 - u) * v * src.getValueByLocation([ob, ihLower, iwUpper, oc]) +
              u * (1 - v) * src.getValueByLocation([ob, ihUpper, iwLower, oc]) +
              u * v * src.getValueByLocation([ob, ihUpper, iwUpper, oc]);
          dst.setValueByLocation([ob, oh, ow, oc], outputValue);
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
    } = {}) {
  if (axes[0] === 0) {
    // hwnc -> nhwc
    input = transpose(input, {permutation: [2, 0, 1, 3]});
  } else if (axes[0] === 2) {
    // nchw -> nhwc
    input = transpose(input, {permutation: [0, 2, 3, 1]});
  }
  let [scaleHeight, scaleWidth] = scales;
  let targetHeight;
  let targetWidth;
  if (sizes !== undefined) {
    [targetHeight, targetWidth] = sizes;
    scaleHeight = targetHeight / input.shape[1];
    scaleWidth = targetWidth / input.shape[2];
  } else {
    targetHeight = Math.floor(input.shape[1] * scaleHeight);
    targetWidth = Math.floor(input.shape[2] * scaleWidth);
  }
  const outputShape = input.shape.slice();
  outputShape[1] = targetHeight;
  outputShape[2] = targetWidth;
  let output = new Tensor(outputShape);
  output = interpolationFunctions[mode](
      input, output, targetHeight, targetWidth, scaleHeight, scaleWidth);
  if (axes[0] === 0) {
    // nhwc -> hwnc
    output = transpose(output, {permutation: [1, 2, 0, 3]});
  } else if (axes[0] === 2) {
    // nhwc -> nchw
    output = transpose(output, {permutation: [0, 3, 1, 2]});
  }
  return output;
}
