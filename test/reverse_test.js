'use strict';

import {reverse} from '../src/reverse.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test reverse', function() {
  function testReverse(inputShape, inputValue, expected, options = {}) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = reverse(inputTensor, options);
    utils.checkShape(outputTensor, inputShape);
    utils.checkValue(outputTensor, expected);
  }

  it('reverse 0D scalar default options', function() {
    testReverse([], [2], [2]);
  });

  it('reverse 4D default options', function() {
    testReverse(
        [2, 2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('reverse 2D with axes=[0]', function() {
    testReverse(
        [2, 2],
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        {
          axes: [0],
        });
  });

  it('reverse 4D with axes=[3]', function() {
    testReverse(
        [2, 2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        {
          axes: [3],
        });
  });

  it('reverse 4D with axes=[1, 2]', function() {
    testReverse(
        [2, 2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [7, 8, 5, 6, 3, 4, 1, 2, 15, 16, 13, 14, 11, 12, 9, 10],
        {
          axes: [1, 2],
        });
  });

  it('reverse 4D with axes=[3, 1, 0, 2]', function() {
    testReverse(
        [2, 2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        {
          axes: [3, 1, 0, 2],
        });
  });
});
