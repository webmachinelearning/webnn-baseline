'use strict';

import {gatherND} from '../src/gather_nd.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test gatherND', function() {
  function testGatherND(input, indices, expected) {
    const inputTensor = new Tensor(input.shape, input.data);
    const indicesTensor = new Tensor(indices.shape, indices.data);
    const outputTensor = gatherND(inputTensor, indicesTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('gatherND with elements from 2D inputs by 2D indices', function() {
    /* eslint-disable max-len */
    // Refer to Example 1 on https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/gather-nd-8.html
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
        0, 0,
        1, 0,
      ],
    };
    const expected = {
      shape: [2],
      data: [1, 3],
    };
    testGatherND(input, indices, expected);
  });

  it('gatherND with slices from 2D inputs by 2D indices', function() {
    /* eslint-disable max-len */
    // Refer to Example 2 on https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/gather-nd-8.html
    const input = {
      shape: [2, 2],
      data: [
        1, 2,
        3, 4,
      ],
    };
    const indices = {
      shape: [2, 1],
      data: [1, 0],
    };
    const expected = {
      shape: [2, 2],
      data: [
        3, 4,
        1, 2,
      ],
    };
    testGatherND(input, indices, expected);
  });

  it('gatherND 2D inputs when 3D indices has leading dimensions', function() {
    /* eslint-disable max-len */
    // Refer to Example 3 on https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/gather-nd-8.html
    const input = {
      shape: [2, 2],
      data: [
        1, 2,
        3, 4,
      ],
    };
    const indices = {
      shape: [2, 1, 1],
      data: [1, 0],
    };
    const expected = {
      shape: [2, 1, 2],
      data: [
        3, 4,
        1, 2,
      ],
    };
    testGatherND(input, indices, expected);
  });

  it('gatherND 2D inputs when 3D negative indices has leading dimensions', function() {
    const input = {
      shape: [2, 2],
      data: [
        1, 2,
        3, 4,
      ],
    };
    const indices = {
      shape: [2, 1, 1],
      data: [-1, -2],
    };
    const expected = {
      shape: [2, 1, 2],
      data: [
        3, 4,
        1, 2,
      ],
    };
    testGatherND(input, indices, expected);
  });

  it('gatherND 3D inputs by 2D indices', function() {
    // Refer to Example 3 on https://onnx.ai/onnx/operators/onnx__GatherND.html
    const input = {
      shape: [2, 2, 2],
      data: [
        0, 1,
        2, 3,
        4, 5,
        6, 7,
      ],
    };
    const indices = {
      shape: [2, 2],
      data: [
        0, 1,
        1, 0,
      ],
    };
    const expected = {
      shape: [2, 2],
      data: [
        2, 3,
        4, 5,
      ],
    };
    testGatherND(input, indices, expected);
  });


  it('gatherND 3D inputs by 3D indices', function() {
    // Refer to Example 4 on https://onnx.ai/onnx/operators/onnx__GatherND.html
    const input = {
      shape: [2, 2, 2],
      data: [
        0, 1,
        2, 3,
        4, 5,
        6, 7,
      ],
    };
    const indices = {
      shape: [2, 1, 2],
      data: [
        0, 1,
        1, 0,
      ],
    };
    const expected = {
      shape: [2, 1, 2],
      data: [
        2, 3,
        4, 5,
      ],
    };
    testGatherND(input, indices, expected);
  });

  it('gatherND 1D inputs by 1D indices', function() {
    const input = {
      shape: [4],
      data: [1, 2, 3, 4],
    };
    const indices = {
      shape: [1],
      data: [1],
    };
    const expected = {
      shape: [],
      data: [2],
    };
    testGatherND(input, indices, expected);
  });

  it('gatherND 4D inputs by 1D indices', function() {
    const input = {
      shape: [2, 2, 2, 2],
      data: [
        1,  2,
        3,  4,
        5,  6,
        7,  8,
        9,  10,
        11, 12,
        13, 14,
        15, 16,
      ],
    };
    const indices = {
      shape: [3],
      data: [1, 0, 0],
    };
    const expected = {
      shape: [2],
      data: [9, 10],
    };
    testGatherND(input, indices, expected);
  });
});
