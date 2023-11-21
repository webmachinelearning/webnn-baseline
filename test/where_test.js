'use strict';

import {where} from '../src/where.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test where', function() {
  function testWhere(condition, inputA, inputB, expected) {
    const tensorCondition = new Tensor(condition.shape, condition.data);
    const tensorA = new Tensor(inputA.shape, inputA.data);
    const tensorB = new Tensor(inputB.shape, inputB.data);
    const outputTensor = where(tensorCondition, tensorA, tensorB);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('where', function() {
    const condition = {
      shape: [2, 3],
      data: [1, 1, 0, 0, 1, 0],
    };
    const inputA = {
      shape: [2, 3],
      data: [1, 2, 3, 4, 5, 64],
    };
    const inputB = {
      shape: [2, 3],
      data: [6, 3, 5, 7, 8, 0],
    };
    const expected = {
      shape: [2, 3],
      data: [1, 2, 5, 7, 5, 0],
    };
    testWhere(condition, inputA, inputB, expected);
  });

  it('where broadcast condition1d×A2d×B2d', function() {
    const condition = {
      shape: [3],
      data: [1, 1, 0],
    };
    const inputA = {
      shape: [2, 3],
      data: [1, 2, 3, 4, 5, 64],
    };
    const inputB = {
      shape: [2, 3],
      data: [7, 8, 9, 10, 11, 12],
    };
    const expected = {
      shape: [2, 3],
      data: [1, 2, 9, 4, 5, 12],
    };
    testWhere(condition, inputA, inputB, expected);
  });

  it('where broadcast condition2d×A2d×B1d', function() {
    const condition = {
      shape: [2, 3],
      data: [1, 1, 0, 0, 0, 1],
    };
    const inputA = {
      shape: [2, 3],
      data: [1, 2, 3, 4, 5, 64],
    };
    const inputB = {
      shape: [3],
      data: [7, 8, 9],
    };
    const expected = {
      shape: [2, 3],
      data: [1, 2, 9, 7, 8, 64],
    };
    testWhere(condition, inputA, inputB, expected);
  });
});
