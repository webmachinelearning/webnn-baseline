'use strict';

import {gatherElements} from '../src/gather_elements.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test gatherElements', function() {
  function testGatherElements(input, indices, expected, options = {}) {
    const inputTensor = new Tensor(input.shape, input.data);
    const indicesTensor = new Tensor(indices.shape, indices.data);
    const outputTensor = gatherElements(inputTensor, indicesTensor, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('gatherElements 2D default', function() {
    const input = {
      shape: [2, 2],
      data: [
        1, 2,
        3, 4,
      ],
    };
    const indices = {
      shape: [2, 2],
      data: [
        0, 1,
        0, 0,
      ],
    };
    const expected = {
      shape: [2, 2],
      data: [
        1, 4,
        1, 2,
      ],
    };
    testGatherElements(input, indices, expected);
  });

  it('gatherElements 2D indices of greater shape, axis=1', function() {
    const input = {
      shape: [2, 2],
      data: [
        1, 7,
        4, 3,
      ],
    };
    const indices = {
      shape: [2, 3],
      data: [
        1, 1, 0,
        1, 0, 1,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        7, 7, 1,
        3, 4, 3,
      ],
    };
    testGatherElements(input, indices, expected, {axis: 1});
  });

  it('gatherElements 2D indices of lesser shape', function() {
    const input = {
      shape: [3, 3],
      data: [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
      ],
    };
    const indices = {
      shape: [2, 3],
      data: [
        1, 0, 1,
        1, 2, 0,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        4, 2, 6,
        4, 8, 3,
      ],
    };
    testGatherElements(input, indices, expected);
  });

  it('gatherElements 2D indices of lesser shape, axis=1', function() {
    const input = {
      shape: [3, 3],
      data: [
        -66.05901336669922, -68.9197006225586, -77.02045440673828,
        -26.158037185668945, 89.0337142944336, -45.89653396606445,
        43.84803771972656, 48.81806945800781, 51.79948425292969,
      ],
    };
    const indices = {
      shape: [3, 2],
      data: [
        1, 0,
        2, 2,
        1, 0,
      ],
    };
    const expected = {
      shape: [3, 2],
      data: [
        -68.9197006225586, -66.05901336669922, -45.89653396606445,
        -45.89653396606445, 48.81806945800781, 43.84803771972656,
      ],
    };
    testGatherElements(input, indices, expected, {axis: 1});
  });

  it('gatherElements 2D negative indices', function() {
    const input = {
      shape: [3, 3],
      data: [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
      ],
    };
    const indices = {
      shape: [2, 3],
      data: [
        -1, -2, 0,
        -2, 0, 0,
      ],
    };
    const expected = {
      shape: [2, 3],
      data: [
        7, 5, 3,
        4, 2, 3,
      ],
    };
    testGatherElements(input, indices, expected);
  });
});
