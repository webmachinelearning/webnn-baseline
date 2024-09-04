'use strict';

import {tile} from '../src/tile.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test tile', function() {
  function testTile(input, repetitions, expected) {
    const tensor = new Tensor(input.shape, input.data);
    const outputTensor = tile(tensor, repetitions);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('tile 1D', function() {
    const input = {
      shape: [4],
      data: [
        1, 2, 3, 4,
      ],
    };
    const repetitions = [2];
    const expected = {
      shape: [8],
      data: [
        1, 2, 3, 4, 1, 2, 3, 4,
      ],
    };
    testTile(input, repetitions, expected);
  });

  it('tile 2D', function() {
    const input = {
      shape: [2, 2],
      data: [
        1, 2,
        3, 4,
      ],
    };
    const repetitions = [2, 3];
    const expected = {
      shape: [4, 6],
      data: [
        1, 2, 1, 2, 1, 2,
        3, 4, 3, 4, 3, 4,
        1, 2, 1, 2, 1, 2,
        3, 4, 3, 4, 3, 4,
      ],
    };
    testTile(input, repetitions, expected);
  });
});
