'use strict';

import {hardSigmoid} from '../src/hard_sigmoid.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test hardSigmoid', function() {
  function testHardSigmoid(inputShape, inputValue, expected, options = {}) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = hardSigmoid(inputTensor, options);
    utils.checkValue(outputTensor, expected);
  }

  it('hardSigmoid default', function() {
    testHardSigmoid(
        [2, 3],
        [
          -1, 0, 1,
          2, 3, 4,
        ],
        [
          0.3, 0.5, 0.7,
          0.9, 1, 1,
        ],
    );
  });

  it('hardSigmoid alpha', function() {
    testHardSigmoid(
        [2, 3],
        [
          -1, 0, 1,
          2, 3, 4,
        ],
        [
          0.25, 0.5, 0.75,
          1, 1, 1,
        ],
        {
          alpha: 0.25,
        },
    );
  });

  it('hardSigmoid beta', function() {
    testHardSigmoid(
        [2, 3],
        [
          -1, 0, 1,
          2, 3, 4,
        ],
        [
          0.04999999999999999, 0.25, 0.45,
          0.65, 0.8500000000000001, 1,
        ],
        {
          beta: 0.25,
        },
    );
  });

  it('hardSigmoid', function() {
    testHardSigmoid(
        [2, 3],
        [
          -1, 0, 1,
          2, 3, 4,
        ],
        [
          0, 0.25, 0.5,
          0.75, 1, 1,
        ],
        {
          alpha: 0.25,
          beta: 0.25,
        },
    );
  });
});
