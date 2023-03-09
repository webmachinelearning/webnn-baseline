'use strict';

import {prelu} from '../src/prelu.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test prelu', function() {
  function testPRelu(input, slope, expected) {
    const inputTensor = new Tensor(input.shape, input.data);
    const slopeTensor = new Tensor(slope.shape, slope.data);
    const outputTensor = prelu(inputTensor, slopeTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('prelu 3d', function() {
    const input = {
      'shape': [1, 2, 3],
      'data': [
        1, -1, 2,
        -2, 3, -3,
      ],
    };
    const slope = {
      'shape': [1, 2, 3],
      'data': [
        0.1, -0.1, 0.25,
        -0.25, -0.5, 0.5,
      ],
    };
    const expected = {
      'shape': [1, 2, 3],
      'data': [
        1, 0.1, 2,
        0.5, 3, -1.5,
      ],
    };
    testPRelu(input, slope, expected);
  });

  it('prelu broadcast 3d x 1d', function() {
    const input = {
      'shape': [1, 2, 3],
      'data': [
        1, -1, 2,
        -2, 3, -3,
      ],
    };
    const slope = {
      'shape': [1],
      'data': [0.1],
    };
    const expected = {
      'shape': [1, 2, 3],
      'data': [
        1, -0.1, 2,
        -0.2, 3, -0.30000000000000004,
      ],
    };
    testPRelu(input, slope, expected);
  });

  it('prelu broadcast 3d x 2d', function() {
    const input = {
      'shape': [1, 2, 3],
      'data': [
        1, -1, 2,
        -2, 3, -3,
      ],
    };
    const slope = {
      'shape': [1, 3],
      'data': [
        0.1, -0.25, 0.5,
      ],
    };
    const expected = {
      'shape': [1, 2, 3],
      'data': [
        1, 0.25, 2,
        -0.2, 3, -1.5,
      ],
    };
    testPRelu(input, slope, expected);
  });

  it('prelu broadcast 3d x 3d', function() {
    const input = {
      'shape': [1, 2, 3],
      'data': [
        1, -1, 2,
        -2, 3, -3,
      ],
    };
    const slope = {
      'shape': [1, 2, 1],
      'data': [
        0.1,
        -0.1,
      ],
    };
    const expected = {
      'shape': [1, 2, 3],
      'data': [
        1, -0.1, 2,
        0.2, 3, 0.30000000000000004,
      ],
    };
    testPRelu(input, slope, expected);
  });
});
