'use strict';

import {squeeze} from '../src/squeeze.js';
import {Tensor, sizeOfShape} from '../src/tensor.js';
import * as utils from './utils.js';

describe('test squeeze', function() {
  function testSqueeze(oldShape, axes, expectedShape) {
    const bufferSize = sizeOfShape(oldShape);
    const inputBuffer = new Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    const x = new Tensor(oldShape, inputBuffer);
    const y = squeeze(x, {axes});
    utils.checkShape(y, expectedShape);
    utils.checkValue(y, inputBuffer);
  }

  it('squeeze one dimension by default', function() {
    testSqueeze([1, 3, 4, 5], undefined, [3, 4, 5]);
  });

  it('squeeze one dimension with axes', function() {
    testSqueeze([1, 3, 1, 5], [0], [3, 1, 5]);
  });

  it('squeeze two dimensions by default', function() {
    testSqueeze([1, 3, 1, 5], undefined, [3, 5]);
  });

  it('squeeze two dimensions with axes', function() {
    testSqueeze([1, 3, 1, 5], [0, 2], [3, 5]);
  });
});
