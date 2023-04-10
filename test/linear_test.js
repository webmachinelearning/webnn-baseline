'use strict';

import {linear} from '../src/linear.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test linear', function() {
  function testLinear(inputShape, inputValue, expected, options = {}) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = linear(inputTensor, options);
    utils.checkValue(outputTensor, expected);
  }

  it('linear default', function() {
    testLinear(
        [3],
        [
          -1, 0, 1,
        ],
        [
          -1, 0, 1,
        ],
    );
  });

  it('linear alpha', function() {
    testLinear(
        [3],
        [
          -1, 0, 1,
        ],
        [
          -0.25, 0, 0.25,
        ],
        {
          alpha: 0.25,
        },
    );
  });

  it('linear beta', function() {
    testLinear(
        [3],
        [
          -1, 0, 1,
        ],
        [
          -0.75, 0.25, 1.25,
        ],
        {
          beta: 0.25,
        },
    );
  });

  it('linear', function() {
    testLinear(
        [3],
        [
          -1, 0, 1,
        ],
        [
          0, 0.25, 0.5,
        ],
        {
          alpha: 0.25,
          beta: 0.25,
        },
    );
  });
});
