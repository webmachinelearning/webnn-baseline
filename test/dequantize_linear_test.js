'use strict';

import {dequantizeLinear} from '../src/dequantize_linear.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test dequantizeLinear', function() {
  function testDequantizeLinear(input, scale, zeroPoint, expected) {
    const inputTensor = new Tensor(input.shape, input.value);
    const scaleTensor = new Tensor(scale.shape, scale.value);
    const zeroPointTensor = new Tensor(zeroPoint.shape, zeroPoint.value);
    const outputTensor = dequantizeLinear(inputTensor, scaleTensor, zeroPointTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.value);
  }


  it('dequantizeLinear 1D broadcasting scale and zeroPoint', function() {
    testDequantizeLinear(
        { // input
          shape: [4],
          value: [0, 3, 128, 255],
        },
        { // scale
          shape: [1],
          value: [2],
        },
        { // zeroPoint of uint8
          shape: [1],
          value: [128],
        },
        { // expected
          shape: [4],
          value: [-256, -250, 0, 254],
        },
    );
  });

  it('dequantizeLinear 2D', function() {
    testDequantizeLinear(
        { // input
          shape: [3, 4],
          value: [
            0, 1,  2,  3,
            0, 1,  2,  3,
            0, 10, 20, 30,
          ],
        },
        { // scale
          shape: [3, 4],
          value: [
            1, 1, 1, 1,
            2, 2, 2, 2,
            4, 4, 4, 4,
          ],
        },
        { // zeroPoint
          shape: [3, 4],
          value: [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
          ],
        },
        { // expected
          shape: [3, 4],
          value: [
            0, 1,  2,  3,
            0, 2,  4,  6,
            0, 40, 80, 120,
          ],
        },
    );
  });

  it('dequantizeLinear 2D broadcasting scale and zeroPoint', function() {
    testDequantizeLinear(
        { // input
          shape: [3, 4],
          value: [
            0, 1,  2,  3,
            0, 1,  2,  3,
            0, 10, 20, 30,
          ],
        },
        { // scale
          shape: [3, 1],
          value: [
            1,
            2,
            4,
          ],
        },
        { // zeroPoint
          shape: [3, 1],
          value: [
            0,
            0,
            0,
          ],
        },
        { // expected
          shape: [3, 4],
          value: [
            0, 1,  2,  3,
            0, 2,  4,  6,
            0, 40, 80, 120,
          ],
        },
    );
  });

  it('dequantizeLinear 4D broadcasting scale and zeroPoint', function() {
    testDequantizeLinear(
        { // input
          shape: [1, 1, 3, 4],
          value: [
            0, 1,  2,  3,
            0, 1,  2,  3,
            0, 10, 20, 30,
          ],
        },
        { // scale
          shape: [3, 1],
          value: [
            1,
            2,
            4,
          ],
        },
        { // zeroPoint
          shape: [1],
          value: [
            0,
          ],
        },
        { // expected
          shape: [1, 1, 3, 4],
          value: [
            0, 1,  2,  3,
            0, 2,  4,  6,
            0, 40, 80, 120,
          ],
        },
    );
  });
});
