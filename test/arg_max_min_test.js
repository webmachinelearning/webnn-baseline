'use strict';

import {argMax, argMin} from '../src/arg_max_min.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test argMax and argMin', function() {
  function testArgMaxMin(input, axis, expected, func, options = {}) {
    const inputTensor = new Tensor(input.shape, input.value);
    const outputTensor = func(inputTensor, axis, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.value);
  }

  it('argMax axis=0', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        0,
        {
          shape: [3],
          value: [1, 2, 1],
        },
        argMax);
  });

  it('argMax axis=1', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        1,
        {
          shape: [3],
          value: [2, 2, 1],
        },
        argMax);
  });

  it('argMax 2D of shape [1, 3] axis=1', function() {
    testArgMaxMin(
        {
          shape: [1, 3],
          value: [1, 2, 3],
        },
        1,
        {
          shape: [1],
          value: [2],
        },
        argMax);
  });

  it('argMax axis=1 keepDimensions=true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 5,
          ],
        },
        1,
        {
          shape: [3, 1],
          value: [2, 2, 1],
        },
        argMax,
        {
          keepDimensions: true,
        });
  });

  it('argMax 3D of shape [1, 3, 4] axis=1 keepDimensions=false', function() {
    testArgMaxMin(
        {
          shape: [1, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 3,
            5, 4, 5, 0,
          ],
        },
        1,
        {
          shape: [1, 4],
          value: [2, 1, 2, 0],
        },
        argMax,
        {
          keepDimensions: false,
        });
  });

  it('argMin axis=0', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        0,
        {
          shape: [3],
          value: [0, 1, 1],
        },
        argMin);
  });

  it('argMin axis=1', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        1,
        {
          shape: [3],
          value: [0, 1, 0],
        },
        argMin);
  });

  it('argMin 2D of shape [1, 3] axis=1', function() {
    testArgMaxMin(
        {
          shape: [1, 3],
          value: [1, 2, 3],
        },
        1,
        {
          shape: [1],
          value: [0],
        },
        argMin);
  });

  it('argMin axis=1 keepDimensions=true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        1,
        {
          shape: [3, 1],
          value: [0, 1, 0],
        },
        argMin,
        {
          keepDimensions: true,
        });
  });
});
