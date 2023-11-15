'use strict';

import {equal, greater, greaterOrEqual, lesser, lesserOrEqual, not} from '../src/logical.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test logical', function() {
  function testLogical(inputA, inputB, expected, func) {
    const tensorA = new Tensor(inputA.shape, inputA.data);
    const tensorB = new Tensor(inputB.shape, inputB.data);
    const outputTensor = func(tensorA, tensorB);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  function testLogicalNot(input, expected) {
    const inputTensor = new Tensor(input.shape, input.data);
    const outputTensor = not(inputTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('equal 0D scalar', function() {
    const inputA = {
      shape: [],
      data: [0.5],
    };
    const inputB = {
      shape: [],
      data: [0.5],
    };
    const expected = {
      shape: [],
      data: [1],
    };
    testLogical(inputA, inputB, expected, equal);
  });

  it('equal 4D', function() {
    const inputA = {
      shape: [1, 2, 2, 1],
      data: [-1, 1, 1, 0],
    };
    const inputB = {
      shape: [1, 2, 2, 1],
      data: [1, 0, -1, 0],
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [0, 0, 0, 1],
    };
    testLogical(inputA, inputB, expected, equal);
  });

  it('equal 4D broadcast', function() {
    const inputA = {
      shape: [1, 2, 2, 1],
      data: [-1, 1, 1, 0],
    };
    const inputB = {
      shape: [1],
      data: [1],
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [0, 1, 1, 0],
    };
    testLogical(inputA, inputB, expected, equal);
  });

  it('greater 0D scalar', function() {
    const inputA = {
      shape: [],
      data: [1],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [1],
    };
    testLogical(inputA, inputB, expected, greater);
  });

  it('greater 4D', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1, 3, 3, 1],
      data: [
        -2, -1,  0,
        -1,  1, -1,
        0,  1,  2,
      ],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        1, 0, 0,
        1, 0, 1,
        1, 0, 0,
      ],
    };
    testLogical(inputA, inputB, expected, greater);
  });

  it('greater 4D broadcast', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1],
      data: [0],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        0, 0, 0,
        0, 0, 1,
        1, 1, 1,
      ],
    };
    testLogical(inputA, inputB, expected, greater);
  });

  it('greaterOrEqual 0D scalar', function() {
    const inputA = {
      shape: [],
      data: [1],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [1],
    };
    testLogical(inputA, inputB, expected, greaterOrEqual);
  });

  it('greaterOrEqual 4D', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1, 3, 3, 1],
      data: [
        -2, -1,  0,
        -1,  1, -1,
        0,  1,  2,
      ],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        1, 1, 0,
        1, 0, 1,
        1, 1, 0,
      ],
    };
    testLogical(inputA, inputB, expected, greaterOrEqual);
  });

  it('greaterOrEqual 4D broadcast', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1],
      data: [0],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        0, 0, 0,
        1, 1, 1,
        1, 1, 1,
      ],
    };
    testLogical(inputA, inputB, expected, greaterOrEqual);
  });

  it('lesser 0D scalar', function() {
    const inputA = {
      shape: [],
      data: [1],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [0],
    };
    testLogical(inputA, inputB, expected, lesser);
  });

  it('lesser 4D', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1, 3, 3, 1],
      data: [
        -2, -1,  0,
        -1,  1, -1,
        0,  1,  2,
      ],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        0, 0, 1,
        0, 1, 0,
        0, 0, 1,
      ],
    };
    testLogical(inputA, inputB, expected, lesser);
  });

  it('lesser 4D broadcast', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1],
      data: [0],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        1, 1, 1,
        0, 0, 0,
        0, 0, 0,
      ],
    };
    testLogical(inputA, inputB, expected, lesser);
  });

  it('lesserOrEqual 0D scalar', function() {
    const inputA = {
      shape: [],
      data: [1],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [0],
    };
    testLogical(inputA, inputB, expected, lesserOrEqual);
  });

  it('lesserOrEqual 4D', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1, 3, 3, 1],
      data: [
        -2, -1,  0,
        -1,  1, -1,
        0,  1,  2,
      ],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        0, 1, 1,
        0, 1, 0,
        0, 1, 1,
      ],
    };
    testLogical(inputA, inputB, expected, lesserOrEqual);
  });

  it('lesserOrEqual 4D broadcast', function() {
    const inputA = {
      shape: [1, 3, 3, 1],
      data: [
        -1, -1, -1,
        0,  0,  1,
        1,  1,  1,
      ],
    };
    const inputB = {
      shape: [1],
      data: [0],
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [
        1, 1, 1,
        1, 1, 0,
        0, 0, 0,
      ],
    };
    testLogical(inputA, inputB, expected, lesserOrEqual);
  });

  it('not 0D scalar', function() {
    const input = {
      shape: [],
      data: [1],
    };
    const expected = {
      shape: [],
      data: [0],
    };
    testLogicalNot(input, expected);
  });


  it('not 4D', function() {
    const input = {
      shape: [1, 2, 2, 1],
      data: [
        0,   1,
        10, 255,
      ],
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [
        1, 0,
        0, 0,
      ],
    };
    testLogicalNot(input, expected);
  });
});
