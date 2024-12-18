'use strict';

import {scatterND} from '../src/scatter_nd.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test scatterND', function() {
  function testscatterND(input, indices, updates, expected) {
    const inputTensor = new Tensor(input.shape, input.data);
    const indicesTensor = new Tensor(indices.shape, indices.data);
    const updatesTensor = new Tensor(updates.shape, updates.data);
    const outputTensor = scatterND(inputTensor, indicesTensor, updatesTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('scatterND to insert individual elements in a tensor by index', function() {
    // Refer to Example 1 on https://onnx.ai/onnx/operators/onnx__ScatterND.html
    const input = {
      shape: [8],
      data: [1, 2, 3, 4, 5, 6, 7, 8],
    };
    const indices = {
      shape: [4, 1],
      data: [4, 3, 1, 7],
    };
    const updates = {
      shape: [4],
      data: [9, 10, 11, 12],
    };
    const expected = {
      shape: [8],
      data: [1, 11, 3, 10, 9, 6, 7, 12],
    };
    testscatterND(input, indices, updates, expected);
  });

  it('scatterND to insert entire slices of a higher rank tensor', function() {
    // Refer to Example 2 on https://onnx.ai/onnx/operators/onnx__ScatterND.html
    const input = {
      shape: [4, 4, 4],
      data: [
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const indices = {
      shape: [2, 1],
      data: [0, 2],
    };
    const updates = {
      shape: [2, 4, 4],
      data: [
        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
      ],
    };
    const expected = {
      shape: [4, 4, 4],
      data: [
        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    testscatterND(input, indices, updates, expected);
  });

  it('scatterND with negative indices', function() {
    const input = {
      shape: [4, 4, 4],
      data: [
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const indices = {
      shape: [2, 1],
      data: [-4, -2],
    };
    const updates = {
      shape: [2, 4, 4],
      data: [
        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
      ],
    };
    const expected = {
      shape: [4, 4, 4],
      data: [
        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
        8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    testscatterND(input, indices, updates, expected);
  });

  it('scatterND with 0D updates', function() {
    const input = {
      shape: [2, 2],
      data: [1, 2, 3, 4],
    };
    const indices = {
      shape: [2],
      data: [1, 0],
    };
    const updates = {
      shape: [],
      data: [100],
    };
    const expected = {
      shape: [2, 2],
      data: [1, 2, 100, 4],
    };
    testscatterND(input, indices, updates, expected);
  });
});
