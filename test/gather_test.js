'use strict';

import {gather} from '../src/gather.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test gather', function() {
  function testGather(input, indices, expected, options = {}) {
    const inputTensor = new Tensor(input.shape, input.data);
    const indicesTensor = new Tensor(indices.shape, indices.data);
    const outputTensor = gather(inputTensor, indicesTensor, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('gather 1D default', function() {
    const input = {
      shape: [4],
      data: [1, 2, 3, 4],
    };
    const indices = {
      shape: [5],
      data: [2, 1, 3, 0, 1],
    };
    const expected = {
      shape: [5],
      data: [3, 2, 4, 1, 2],
    };
    testGather(input, indices, expected);
  });

  it('gather 1D by 0D indices default', function() {
    const input = {
      shape: [4],
      data: [1, 2, 3, 4],
    };
    const indices = {
      shape: [],
      data: [2],
    };
    const expected = {
      shape: [],
      data: [3],
    };
    testGather(input, indices, expected);
  });

  it('gather 2D default', function() {
    const input = {
      shape: [4, 3],
      data: [
        0,  1,  2,
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
      ],
    };
    const indices = {
      shape: [2],
      data: [3, 1],
    };
    const expected = {
      shape: [2, 3],
      data: [
        30, 31, 32,
        10, 11, 12,
      ],
    };
    testGather(input, indices, expected);
  });

  it('gather explicit axis=0', function() {
    const input = {
      shape: [4, 3],
      data: [
        0,  1,  2,
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
      ],
    };
    const indices = {
      shape: [2],
      data: [3, 1],
    };
    const expected = {
      shape: [2, 3],
      data: [
        30, 31, 32,
        10, 11, 12,
      ],
    };
    testGather(input, indices, expected, {axis: 0});
  });

  it('gather 2D axis=1', function() {
    const input = {
      shape: [4, 3],
      data: [
        0,  1,  2,
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
      ],
    };
    const indices = {
      shape: [3],
      data: [2, 1, 1],
    };
    const expected = {
      shape: [4, 3],
      data: [
        2,  1,  1,
        12, 11, 11,
        22, 21, 21,
        32, 31, 31,
      ],
    };
    testGather(input, indices, expected, {axis: 1});
  });

  it('gather 2D by 0D indices axis=1', function() {
    const input = {
      shape: [4, 3],
      data: [
        0,  1,  2,
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
      ],
    };
    const indices = {
      shape: [],
      data: [1],
    };
    const expected = {
      shape: [4],
      data: [
        1, 11, 21, 31,
      ],
    };
    testGather(input, indices, expected, {axis: 1});
  });

  it('gather 2D by 2D indices axis=1', function() {
    const input = {
      shape: [4, 3],
      data: [
        0,  1,  2,
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
      ],
    };
    const indices = {
      shape: [2, 2],
      data: [0, 1, 1, 2],
    };
    const expected = {
      shape: [4, 2, 2],
      data: [
        0,  1,  1, 2,
        10, 11, 11, 12,
        20, 21, 21, 22,
        30, 31, 31, 32,
      ],
    };
    testGather(input, indices, expected, {axis: 1});
  });

  it('gather 3D axis=0', function() {
    const input = {
      shape: [2, 4, 3],
      data: [
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    const indices = {
      shape: [3],
      data: [1, 0, 1],
    };
    const expected = {
      shape: [3, 4, 3],
      data: [
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    testGather(input, indices, expected, {axis: 0});
  });

  it('gather 3D by 0D indices axis=0', function() {
    const input = {
      shape: [2, 4, 3],
      data: [
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    const indices = {
      shape: [],
      data: [1],
    };
    const expected = {
      shape: [4, 3],
      data: [
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    testGather(input, indices, expected, {axis: 0});
  });

  it('gather 3D by 2D indices axis=0', function() {
    const input = {
      shape: [2, 4, 3],
      data: [
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    const indices = {
      shape: [3, 2],
      data: [1, 0, 1, 0, 0, 1],
    };
    const expected = {
      shape: [3, 2, 4, 3],
      data: [
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    testGather(input, indices, expected, {axis: 0});
  });

  it('gather 3D by axis=1', function() {
    const input = {
      shape: [2, 4, 3],
      data: [
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    const indices = {
      shape: [2],
      data: [2, 0],
    };
    const expected = {
      shape: [2, 2, 3],
      data: [
        20,  21,  22,
        0,   1,   2,
        120, 121, 122,
        100, 101, 102,
      ],
    };
    testGather(input, indices, expected, {axis: 1});
  });

  it('gather 3D by 2D indices axis=2', function() {
    const input = {
      shape: [2, 4, 3],
      data: [
        0,   1,   2,
        10,  11,  12,
        20,  21,  22,
        30,  31,  32,
        100, 101, 102,
        110, 111, 112,
        120, 121, 122,
        130, 131, 132,
      ],
    };
    const indices = {
      shape: [2, 2],
      data: [0, 1, 1, 2],
    };
    const expected = {
      shape: [2, 4, 2, 2],
      data: [
        0,   1,   1,   2,
        10,  11,  11,  12,
        20,  21,  21,  22,
        30,  31,  31,  32,
        100, 101, 101, 102,
        110, 111, 111, 112,
        120, 121, 121, 122,
        130, 131, 131, 132,
      ],
    };
    testGather(input, indices, expected, {axis: 2});
  });
});
