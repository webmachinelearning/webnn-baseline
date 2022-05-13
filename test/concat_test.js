'use strict';

import {concat} from '../src/concat.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test concat', function() {
  function testConcat(tensors, expected) {
    const inputs = [];
    for (let i = 0; i < tensors.length; i++) {
      inputs.push(new Tensor(tensors[i].shape, tensors[i].value));
    }
    const output = concat(inputs, expected.axis);
    utils.checkShape(output, expected.shape);
    utils.checkValue(output, expected.value);
  }

  it('concat 1d', function() {
    const tensors = [
      {shape: [2], value: [1, 2]},
      {shape: [2], value: [3, 4]},
    ];
    const expected = {axis: 0, shape: [4], value: [1, 2, 3, 4]};
    testConcat(tensors, expected);
  });

  it('concat 2d axis=0', function() {
    const tensors = [
      {shape: [1, 2], value: [1, 2]},
      {shape: [2, 2], value: [3, 4, 5, 6]},
    ];
    const expected = {axis: 0, shape: [3, 2], value: [1, 2, 3, 4, 5, 6]};
    testConcat(tensors, expected);
  });

  it('concat 2d axis=1', function() {
    const tensors = [
      {shape: [2, 1], value: [1, 2]},
      {shape: [2, 2], value: [3, 4, 5, 6]},
    ];
    const expected = {axis: 1, shape: [2, 3], value: [1, 3, 4, 2, 5, 6]};
    testConcat(tensors, expected);
  });

  it('concat 2d', function() {
    const tensors = [
      {shape: [2, 2], value: [1, 2, 3, 4]},
      {shape: [2, 2], value: [5, 6, 7, 8]},
    ];
    const expected = [
      {axis: 0, shape: [4, 2], value: [1, 2, 3, 4, 5, 6, 7, 8]},
      {axis: 1, shape: [2, 4], value: [1, 2, 5, 6, 3, 4, 7, 8]},
    ];
    for (const test of expected) {
      testConcat(tensors, test);
    }
  });

  it('concat 3d', function() {
    const tensors = [
      {
        shape: [2, 2, 2],
        value: [1, 2, 3, 4, 5, 6, 7, 8],
      },
      {
        shape: [2, 2, 2],
        value: [9, 10, 11, 12, 13, 14, 15, 16],
      },
    ];
    const expected = [
      {
        axis: 0,
        shape: [4, 2, 2],
        value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      },
      {
        axis: 1,
        shape: [2, 4, 2],
        value: [1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16],
      },
      {
        axis: 2,
        shape: [2, 2, 4],
        value: [1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16],
      },
    ];
    for (const test of expected) {
      testConcat(tensors, test);
    }
  });
});
