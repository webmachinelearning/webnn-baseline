'use strict';

import {expand} from '../src/expand.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test expand', function() {
  function testExpand(input, newShape, expected) {
    const tensor = new Tensor(input.shape, input.data);
    const outputTensor = expand(tensor, newShape);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('expand changed dimensions', function() {
    const input = {
      shape: [2, 3],
      data: [
        1, 1, 0,
        0, 1, 0,
      ],
    };

    const newShape = [2, 2, 3];

    const expected = {
      shape: [2, 2, 3],
      data: [
        1, 1, 0,
        0, 1, 0,
        1, 1, 0,
        0, 1, 0,
      ],
    };
    testExpand(input, newShape, expected);
  });

  it('expand unchanged dimensions', function() {
    const input = {
      shape: [3, 1],
      data: [
        1, 2, 3,
      ],
    };

    const newShape = [3, 4];

    const expected = {
      shape: [3, 4],
      data: [
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
      ],
    };
    testExpand(input, newShape, expected);
  });
});
