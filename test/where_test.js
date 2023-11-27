'use strict';

import {where} from '../src/where.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test where', function() {
  function testWhere(condition, trueValues, falseValues, expected) {
    const tensorCondition = new Tensor(condition.shape, condition.data);
    const tensorA = new Tensor(trueValues.shape, trueValues.data);
    const tensorB = new Tensor(falseValues.shape, falseValues.data);
    const outputTensor = where(tensorCondition, tensorA, tensorB);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('where', function() {
    const condition = {
      shape: [2, 3],
      data: [
        1, 1, 0,
        0, 1, 0,
      ],
    };
    const trueValues = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 64,
      ],
    };
    const falseValues = {
      shape: [2, 3],
      data: [
        6, 3, 5,
        7, 8, 0,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        1, 2, 5,
        7, 5, 0,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });


  it('where broadcast condition1d×A2d×B2d', function() {
    const condition = {
      shape: [3],
      data: [1, 1, 0],
    };
    const trueValues = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 64,
      ],
    };
    const falseValues = {
      shape: [2, 3],
      data: [
        7, 8, 9,
        10, 11, 12,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        1, 2, 9,
        4, 5, 12,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });

  it('where broadcast condition2d×A2d×B1d', function() {
    const condition = {
      shape: [2, 3],
      data: [
        1, 1, 0,
        0, 0, 1,
      ],
    };
    const trueValues = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 64,
      ],
    };
    const falseValues = {
      shape: [3],
      data: [7, 8, 9],
    };
    const expected = {
      shape: [2, 3],
      data: [
        1, 2, 9,
        7, 8, 64,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });

  it('where broadcast condition1d×A2d×B3d', function() {
    const condition = {
      shape: [3],
      data: [
        1, 1, 0,
      ],
    };
    const trueValues = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 64,
      ],
    };
    const falseValues = {
      shape: [2, 2, 3],
      data: [
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18],
    };
    const expected = {
      shape: [2, 2, 3],
      data: [
        1, 2, 9, 4, 5, 12,
        1, 2, 15, 4, 5, 18,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });

  it('where broadcast condition3d×A2d×B1d', function() {
    const condition = {
      shape: [2, 2, 3],
      data: [
        1, 1, 0, 1, 1, 0,
        1, 1, 0, 1, 1, 0,
      ],
    };
    const trueValues = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 64,
      ],
    };
    const falseValues = {
      shape: [3],
      data: [
        7, 8, 9,
      ],
    };
    const expected = {
      shape: [2, 2, 3],
      data: [
        1, 2, 9, 4, 5, 9,
        1, 2, 9, 4, 5, 9,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });

  it('where broadcast !0=true test', function() {
    const condition = {
      shape: [2, 3],
      data: [
        2, 3, 0,
        0, 5, 0,
      ],
    };
    const trueValues = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 64,
      ],
    };
    const falseValues = {
      shape: [2, 3],
      data: [
        6, 3, 5,
        7, 8, 0,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        1, 2, 5,
        7, 5, 0,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });

  it('where broadcast condition2d×A(scalar)×B1d test', function() {
    const condition = {
      shape: [2, 3],
      data: [
        1, 1, 0,
        0, 1, 0,
      ],
    };
    const trueValues = {
      shape: [],
      data: [
        6,
      ],
    };
    const falseValues = {
      shape: [2, 3],
      data: [
        6, 3, 5,
        7, 8, 0,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        6, 6, 5,
        7, 6, 0,
      ],
    };
    testWhere(condition, trueValues, falseValues, expected);
  });
});
