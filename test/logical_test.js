'use strict';

import {equal, greater, greaterOrEqual, lesser, lesserOrEqual,
  logicalAnd, logicalNot, logicalOr, logicalXor, notEqual} from '../src/logical.js';
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
    const outputTensor = logicalNot(inputTensor);
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

  it('logicalAnd 0D uint8 scalar', function() {
    const inputA = {
      shape: [],
      data: [255],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [0],
    };
    testLogical(inputA, inputB, expected, logicalAnd);
  });

  it('logicalAnd 4D uint8 tensors', function() {
    const inputA = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 128, 255, 0, 10, 100, 200],
    };
    const inputB = {
      shape: [2, 2, 2, 1],
      data: [200, 100, 10, 0, 200, 100, 10, 0],
    };
    const expected = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 1, 0, 0, 1, 1, 0],
    };
    testLogical(inputA, inputB, expected, logicalAnd);
  });

  it('logicalAnd 4D uint8 tensors broadcast', function() {
    const inputA = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 128, 255, 0, 10, 100, 200],
    };
    const inputB = {
      shape: [2, 2, 1],
      data: [200, 100, 10, 0],
    };
    const expected = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 1, 0, 0, 1, 1, 0],
    };
    testLogical(inputA, inputB, expected, logicalAnd);
  });

  it('logicalNot 0D scalar', function() {
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


  it('logicalNot 4D', function() {
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

  it('logicalOR 0D uint8 scalar', function() {
    const inputA = {
      shape: [],
      data: [255],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [1],
    };
    testLogical(inputA, inputB, expected, logicalOr);
  });

  it('logicalOr 4D uint8 tensors', function() {
    const inputA = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 128, 255, 0, 10, 100, 200],
    };
    const inputB = {
      shape: [2, 2, 2, 1],
      data: [0, 0, 10, 0, 100, 100, 20, 255],
    };
    const expected = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 1, 1, 1, 1, 1, 1],
    };
    testLogical(inputA, inputB, expected, logicalOr);
  });

  it('logicalOr 4D uint8 tensors broadcast', function() {
    const inputA = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 128, 255, 0, 10, 100, 200],
    };
    const inputB = {
      shape: [2, 2, 1],
      data: [0, 100, 10, 0],
    };
    const expected = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 1, 1, 0, 1, 1, 1],
    };
    testLogical(inputA, inputB, expected, logicalOr);
  });

  it('logicalXor 0D uint8 scalar', function() {
    const inputA = {
      shape: [],
      data: [255],
    };
    const inputB = {
      shape: [],
      data: [0],
    };
    const expected = {
      shape: [],
      data: [1],
    };
    testLogical(inputA, inputB, expected, logicalXor);
  });

  it('logicalXor 4D uint8 tensors', function() {
    const inputA = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 128, 255, 0, 10, 100, 200],
    };
    const inputB = {
      shape: [2, 2, 2, 1],
      data: [0, 0, 10, 0, 100, 100, 20, 255],
    };
    const expected = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 0, 1, 1, 0, 0, 0],
    };
    testLogical(inputA, inputB, expected, logicalXor);
  });

  it('logicalXor 4D uint8 tensors broadcast', function() {
    const inputA = {
      shape: [2, 2, 2, 1],
      data: [0, 1, 128, 255, 0, 10, 100, 200],
    };
    const inputB = {
      shape: [2, 2, 1],
      data: [0, 100, 10, 0],
    };
    const expected = {
      shape: [2, 2, 2, 1],
      data: [0, 0, 0, 1, 0, 0, 0, 1],
    };
    testLogical(inputA, inputB, expected, logicalXor);
  });

  it('notEqual 0D scalar', function() {
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
      data: [0],
    };
    testLogical(inputA, inputB, expected, notEqual);
  });

  it('notEqual 4D', function() {
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
      data: [1, 1, 1, 0],
    };
    testLogical(inputA, inputB, expected, notEqual);
  });

  it('notEqual 4D broadcast', function() {
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
      data: [1, 0, 0, 1],
    };
    testLogical(inputA, inputB, expected, notEqual);
  });
});
