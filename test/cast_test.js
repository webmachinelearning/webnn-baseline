'use strict';

import {Tensor} from '../src/lib/tensor.js';
import {cast} from '../src/cast.js';
import * as utils from './utils.js';

describe('test cast', function() {
  function testCast(input, type, expected) {
    const tensorInput = new Tensor(input.shape, input.data);
    const outputTensor = cast(tensorInput, type);
    console.log('output', outputTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('cast float32', function() {
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

  it('cast int32', function() {
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


  it('cast uint32', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.21, 1234, -1234,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0, 0, 3, 1234, 4294966062,
      ],
    };
    testCast(input, 'uint32', expected);
  });

  it('cast int64', function() {
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

  it('cast int8', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.21, 1234, -1234,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0, 0, 3, -46, 46,
      ],
    };
    testCast(input, 'int8', expected);
  });

  it('cast uint8', function() {
    const input = {
      shape: [5],
      data: [
        -0.25, 0.25, 3.21, 1234, -1234,
      ],
    };
    const expected = {
      shape: [5],
      data: [
        0, 0, 3, 210, 46,
      ],
    };
    testCast(input, 'uint8', expected);
  });
});
