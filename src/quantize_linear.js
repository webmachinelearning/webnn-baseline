'use strict';

import {add, div} from './binary.js';
import {clamp} from './clamp.js';
import {unary} from './unary.js';

function roundToNearestEvens(x) {
  return Math.floor(x) % 2 == 0 ? Math.floor(x) : Math.ceil(x);
}

/**
 * Elementwise operator to quantize floating point down to a low precision integer
 * (typically uint8 with a zero-point bias).
 * The calculation follows the expression roundToNearestEvens(input / scale) + zeroPoint.
 * @param {Tensor} input
 * @param {Tensor} scale
 * @param {Tensor} zeroPoint
 * @param {String} dataType
 * @return {Tensor}
 */
export function quantizeLinear(input, scale, zeroPoint, dataType) {
  const addOutput = add(div(input, scale), zeroPoint);
  const roundOutput = unary(addOutput, (x) => roundToNearestEvens(x));

  let maxValue; let minValue;
  switch (dataType) {
    case 'uint8':
      // uint8: [0, 255]
      maxValue = 255;
      minValue = 0;
      break;
    case 'int8':
      // int8: [-128, 127]
      maxValue = 127;
      minValue = -128;
      break;
    case 'uint4':
      // uint4: [0, 15]
      maxValue = 15;
      minValue = 0;
      break;
    case 'int4':
      // int4: [-8, 7]
      maxValue = 7;
      minValue = -8;
      break;
    default:
      throw new Error(`Unsupported ${dataType} data type`);
  }
  return clamp(roundOutput, {minValue, maxValue});
}
