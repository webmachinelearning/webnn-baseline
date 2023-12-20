'use strict';

import {constant} from '../src/constant.js';
import * as utils from './utils.js';


describe('test constant', function() {
  function testConstant(start, step, outputShape, type, expected) {
    const outputTensor = constant(start, step, outputShape, type);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('constant step > 0', function() {
    const expected ={
      shape: [3, 3],
      data: [
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
      ],
    };
    testConstant(
        0, 1, [3, 3], 'float32', expected);
  });

  it('constant step < 0', function() {
    const expected ={
      shape: [3, 3],
      data: [
        9, 8, 7,
        6, 5, 4,
        3, 2, 1,
      ],
    };
    testConstant(
        9, -1, [3, 3], 'float32', expected);
  });

  it('constant step = 0', function() {
    const expected ={
      shape: [3, 3],
      data: [
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
      ],
    };
    testConstant(
        1, 0, [3, 3], 'float64', expected);
  });
});
