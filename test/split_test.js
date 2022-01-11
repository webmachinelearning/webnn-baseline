'use strict';

import {split} from '../src/split.js';
import {Tensor} from '../src/tensor.js';
import * as utils from './utils.js';

describe('test split', function() {
  function testSplit(
      inputShape, inputValue, expectedArray, splits, axis = undefined) {
    const x = new Tensor(inputShape, inputValue);
    const options = {};
    if (axis !== undefined) {
      options.axis = axis;
    }
    const splittedOutputs = split(x, splits, options);
    for (let i = 0; i < splittedOutputs.length; ++i) {
      utils.checkShape(splittedOutputs[i], expectedArray[i].shape);
      utils.checkValue(splittedOutputs[i], expectedArray[i].value);
    }
  }

  it('split', function() {
    testSplit(
        [6], [1, 2, 3, 4, 5, 6],
        [
          {shape: [2], value: [1, 2]},
          {shape: [2], value: [3, 4]},
          {shape: [2], value: [5, 6]},
        ],
        3);
    testSplit(
        [6], [1, 2, 3, 4, 5, 6],
        [{shape: [2], value: [1, 2]}, {shape: [4], value: [3, 4, 5, 6]}],
        [2, 4]);
  });

  it('split 2d', function() {
    testSplit(
        [2, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [
          {shape: [2, 3], value: [1, 2, 3, 7, 8, 9]},
          {shape: [2, 3], value: [4, 5, 6, 10, 11, 12]},
        ],
        2, 1);
    testSplit(
        [2, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [
          {shape: [2, 2], value: [1, 2, 7, 8]},
          {shape: [2, 4], value: [3, 4, 5, 6, 9, 10, 11, 12]},
        ],
        [2, 4], 1);
  });
});
