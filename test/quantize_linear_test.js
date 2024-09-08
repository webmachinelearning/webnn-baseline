'use strict';

import {quantizeLinear} from '../src/quantize_linear.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test quantizeLinear', function() {
  function testQuantizeLinear(input, scale, zeroPoint, dataType, expected) {
    const inputTensor = new Tensor(input.shape, input.value);
    const scaleTensor = new Tensor(scale.shape, scale.value);
    const zeroPointTensor = new Tensor(zeroPoint.shape, zeroPoint.value);
    const outputTensor = quantizeLinear(inputTensor, scaleTensor, zeroPointTensor, dataType);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.value);
  }

  it('quantizeLinear 0D', function() {
    testQuantizeLinear(
        { // input
          shape: [],
          value: [1000],
        },
        { // scale
          shape: [],
          value: [2],
        },
        { // zeroPoint
          shape: [],
          value: [128],
        },
        'uint8',
        { // expected
          shape: [],
          value: [255],
        },
    );
  });

  it('quantizeLinear 1D to uint8 broadcasting scale and zeroPoint', function() {
    testQuantizeLinear(
        { // input
          shape: [6],
          value: [0, 2, 3, 1000, -254, -1000],
        },
        { // scale
          shape: [1],
          value: [2],
        },
        { // zeroPoint
          shape: [1],
          value: [128],
        },
        'uint8',
        { // expected
          shape: [6],
          value: [128, 129, 130, 255, 1, 0],
        },
    );
  });

  it('quantizeLinear 2D to uint8', function() {
    testQuantizeLinear(
        { // input
          shape: [3, 4],
          value: [
            0, 2, 3, 1000,
            0, 2, 3, 1000,
            0, 2, 3, 1000,

          ],
        },
        { // scale
          shape: [3, 4],
          value: [
            1, 1, 1, 1,
            2, 2, 2, 2,
            5, 5, 5, 5,
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
        'uint8',
        { // expected
          shape: [3, 4],
          value: [
            0, 2, 3, 255,
            0, 1, 2, 255,
            0, 0, 0, 200,
          ],
        },
    );
  });

  it('quantizeLinear 4D to int8 broadcasting scale and zeroPoint', function() {
    testQuantizeLinear(
        { // input
          shape: [1, 1, 2, 3],
          value: [
            0,    2,   3,
            1000, -10, -1000,
          ],
        },
        { // scale
          shape: [2, 1],
          value: [
            2,
            2,
          ],
        },
        { // zeroPoint
          shape: [1],
          value: [
            10,
          ],
        },
        'int8',
        { // expected
          shape: [1, 1, 2, 3],
          value: [
            10,  11, 12,
            127, 5,  -128,
          ],
        },
    );
  });

  it('quantizeLinear 4D to int8 rounding halves toward nearest even', function() {
    testQuantizeLinear(
        { // input
          shape: [1, 1, 1, 11],
          value: [
            0, 0.5, 1, 1.5, 2, 2.5, -0.5, -1, -1.5, -2, -2.5,
          ],
        },
        { // scale
          shape: [1],
          value: [1],
        },
        { // zeroPoint
          shape: [1],
          value: [100],
        },
        'int8',
        { // expected
          shape: [1, 1, 1, 11],
          value: [100, 100, 101, 102, 102, 102, 100, 99, 98, 98, 98],
        },
    );
  });
});
