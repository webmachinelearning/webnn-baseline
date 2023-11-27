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

  it('expand a 3D input with a 4D newShape to a 4D output.', function() {
    const input = {
      shape: [2, 1, 4],
      data: [
        1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };

    const newShape = [5, 1, 3, 4];

    const expected = {
      shape: [5, 2, 3, 4],
      data: [
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
      ],
    };
    testExpand(input, newShape, expected);
  });

  it('expand a 3D input with a 2D newShape to a 3D output', function() {
    const input = {
      shape: [2, 1, 4],
      data: [
        1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };

    const newShape = [3, 1];

    const expected = {
      shape: [2, 3, 4],
      data: [
        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
      ],
    };
    testExpand(input, newShape, expected);
  });

  it('expand a 2D input with a 2D newShape to a 2D output', function() {
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

  it('expand a 0D input with a 2D newShape to a 2D output.', function() {
    const input = {
      shape: [],
      data: [
        6,
      ],
    };

    const newShape = [2, 3];

    const expected = {
      shape: [2, 3],
      data: [
        6, 6, 6,
        6, 6, 6,
      ],
    };
    testExpand(input, newShape, expected);
  });

  it('expand a 2D input with a 0D newShape to a 2D output.', function() {
    const input = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 6,
      ],
    };

    const newShape = [];

    const expected = {
      shape: [2, 3],
      data: [
        1, 2, 3,
        4, 5, 6,
      ],
    };
    testExpand(input, newShape, expected);
  });
});
