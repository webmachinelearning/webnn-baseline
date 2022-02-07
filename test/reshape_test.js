'use strict';

import {reshape} from '../src/reshape.js';
import {Tensor, sizeOfShape} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test reshape', function() {
  function testReshape(oldShape, newShape, expectedShape) {
    const bufferSize = sizeOfShape(oldShape);
    const inputBuffer = new Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    const x = new Tensor(oldShape, inputBuffer);
    const y = reshape(x, newShape);
    utils.checkShape(y, expectedShape ? expectedShape : newShape);
    utils.checkValue(y, inputBuffer);
  }

  it('reshape reordered_all_dims', function() {
    testReshape([2, 3, 4], [4, 2, 3]);
  });

  it('reshape reordered_last_dims', function() {
    testReshape([2, 3, 4], [2, 4, 3]);
  });

  it('reshape reduced_dims', function() {
    testReshape([2, 3, 4], [2, 12]);
  });

  it('reshape extended_dims', function() {
    testReshape([2, 3, 4], [2, 3, 2, 2]);
  });

  it('reshape one_dim', function() {
    testReshape([2, 3, 4], [24]);
  });

  it('reshape [2, 3, 4] to negative_dim [2, -1, 2]', function() {
    testReshape([2, 3, 4], [2, -1, 2], [2, 6, 2]);
  });

  it('reshape [2, 3, 4] to negative_dim [-1, 2, 3, 4]', function() {
    testReshape([2, 3, 4], [-1, 2, 3, 4], [1, 2, 3, 4]);
  });
});
