'use strict';

import {scatterElements} from '../src/scatter_elements.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test scatterElements', function() {
  function testScatterElements(input, indices, updates, expected, options = {}) {
    const inputTensor = new Tensor(input.shape, input.data);
    const indicesTensor = new Tensor(indices.shape, indices.data);
    const updatesTensor = new Tensor(updates.shape, updates.data);
    const outputTensor = scatterElements(inputTensor, indicesTensor, updatesTensor, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('scatterElements 2D default', function() {
    const input = {
      shape: [3, 3],
      data: [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
      ],
    };
    const indices = {
      shape: [2, 3],
      data: [
        1, 0, 2,
        0, 2, 1,
      ],
    };
    const updates = {
      shape: [2, 3],
      data: [
        1, 1.1, 1.2,
        2, 2.1, 2.2,
      ],
    };
    const expected = {
      shape: [3, 3],
      data: [
        2, 1.1, 0,
        1, 0,   2.2,
        0, 2.1, 1.2,
      ],
    };
    testScatterElements(input, indices, updates, expected);
  });

  it('scatterElements 2D, explicit axis=0', function() {
    const input = {
      shape: [3, 3],
      data: [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
      ],
    };
    const indices = {
      shape: [2, 3],
      data: [
        1, 0, 2,
        0, 2, 1,
      ],
    };
    const updates = {
      shape: [2, 3],
      data: [
        1, 1.1, 1.2,
        2, 2.1, 2.2,
      ],
    };
    const expected = {
      shape: [3, 3],
      data: [
        2, 1.1, 0,
        1, 0,   2.2,
        0, 2.1, 1.2,
      ],
    };
    testScatterElements(input, indices, updates, expected, {axis: 0});
  });

  it('scatterElements 2D, axis=1', function() {
    const input = {
      shape: [1, 5],
      data: [1, 2, 3, 4, 5],
    };
    const indices = {
      shape: [1, 2],
      data: [1, 3],
    };
    const updates = {
      shape: [1, 2],
      data: [1.1, 2.1],
    };
    const expected = {
      shape: [1, 5],
      data: [1, 1.1, 3, 2.1, 5],
    };
    testScatterElements(input, indices, updates, expected, {axis: 1});
  });

  it('scatterElements 2D negative indices, axis=1', function() {
    const input = {
      shape: [1, 5],
      data: [1, 2, 3, 4, 5],
    };
    const indices = {
      shape: [1, 2],
      data: [1, -2],
    };
    const updates = {
      shape: [1, 2],
      data: [1.1, 2.1],
    };
    const expected = {
      shape: [1, 5],
      data: [1, 1.1, 3, 2.1, 5],
    };
    testScatterElements(input, indices, updates, expected, {axis: 1});
  });

  it('scatterElements throws error with overlapping indices', function() {
    const input = {
      shape: [1, 5],
      data: [1, 2, 3, 4, 5],
    };
    const indices = {
      shape: [1, 2],
      data: [3, -2],
    };
    const updates = {
      shape: [1, 2],
      data: [1.1, 2.1],
    };
    const expected = {
      shape: [1, 5],
      data: [1, 2, 3, 2.1, 5],
    };
    utils.expectThrowError(() => {
      testScatterElements(input, indices, updates, expected, {axis: 1});
    }, 'Invalid indices, [0,-2] is not unique.');
  });
});
