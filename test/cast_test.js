'use strict';

import {Tensor} from '../src/lib/tensor.js';
import {cast} from '../src/cast.js';
import * as utils from './utils.js';

describe('test cast', function() {
  const InputDataType = {
    int8: Int8Array,
    uint8: Uint8Array,
    int32: Int32Array,
    uint32: Uint32Array,
    int64: BigInt64Array,
    float32: Float32Array,
    float64: Float64Array,
  };
  function testCast(input, type, expected) {
    let tensorInput;
    if (input.type) {
      tensorInput = new Tensor(input.shape, new InputDataType[input.type](input.data));
    } else {
      tensorInput = new Tensor(input.shape, input.data);
    }
    const outputTensor = cast(tensorInput, type);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('cast float64 to int8', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.75, 14, -14,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0, 0, 4, 14, -14,
      ],
    };
    testCast(input, 'int8', expected);
  });

  it('cast float64 to uint8', function() {
    const input = {
      shape: [5],
      data: [
        0.25, 0.75, 3.75, 14, 15,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0, 1, 4, 14, 15,
      ],
    };
    testCast(input, 'uint8', expected);
  });

  it('cast float64 to int32', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.21, 1234, -1234,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0, 0, 3, 1234, -1234,
      ],
    };
    testCast(input, 'int32', expected);
  });

  it('cast float64 to uint32', function() {
    const input = {
      shape: [5],
      data: [
        0.75, 0.25, 3.21, 14, 15,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        1, 0, 3, 14, 15,
      ],
    };
    testCast(input, 'uint32', expected);
  });

  it('cast float64 to int64', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.21, 1234, -1234,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0n, 0n, 3n, 1234n, -1234n,
      ],
    };
    testCast(input, 'int64', expected);
  });

  it('cast float64 to float32', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.21, 1234, -1234,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.2100000381469727, 1234, -1234,
      ],
    };
    testCast(input, 'float32', expected);
  });

  it('cast int32 to float32', function() {
    const input = {
      shape: [5],
      data: [
        0, 1, -2, -3, 3,
      ],
      type: 'int32',
    };
    const expected = {
      shape: [5],
      data: [
        0, 1, -2, -3, 3,
      ],
    };
    testCast(input, 'float32', expected);
  });

  it('cast uint32 to float64', function() {
    const input = {
      shape: [5],
      data: [
        0, 1, 22, 33, 33,
      ],
      type: 'uint32',
    };
    const expected = {
      shape: [5],
      data: [
        0, 1, 22, 33, 33,
      ],
    };
    testCast(input, 'float64', expected);
  });

  it('cast float32 to float64', function() {
    const input = {
      shape: [5],
      data: [
        0, 0.1, 0.2, -3, 993,
      ],
      type: 'float32',
    };
    const expected = {
      shape: [5],
      data: [
        0, 0.10000000149011612, 0.20000000298023224, -3, 993,
      ],
    };
    testCast(input, 'float64', expected);
  });
});
