'use strict';

import {convTranspose2d} from '../src/conv_transpose2d.js';
import {clamp} from '../src/clamp.js';
import {leakyRelu} from '../src/leaky_relu.js';
import {relu} from '../src/relu.js';
import {sigmoid} from '../src/sigmoid.js';
import {Tensor} from '../src/lib/tensor.js';

import * as utils from './utils.js';

describe('test convTranspose2d', function() {
  function testConvTranspose2d(
      input, filter, expected, options = {}, bias = undefined,
      activation = undefined, fusion = false, activationOptions = {}) {
    const inputTensor = new Tensor(input.shape, input.data);
    const filterTensor = new Tensor(filter.shape, filter.data);
    if (bias) {
      options.bias = new Tensor(bias.shape, bias.data);
    }
    if (activation === 'relu') {
      options.activation = relu;
    } else if (activation === 'relu6') {
      options.activation = utils.bindTrailingArgs(clamp, {minValue: 0, maxValue: 6});
    } else if (activation === 'sigmoid') {
      options.activation = sigmoid;
    } else if (activation === 'leakyRelu') {
      options.activation = utils.bindTrailingArgs(leakyRelu, activationOptions);
    }

    const outputTensor = convTranspose2d(inputTensor, filterTensor, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('convTranspose2d default options', function() {
    const input = {
      shape: [1, 1, 2, 2],
      data: [
        0,  1,  2,  3,
      ],
    };
    const filter = {
      shape: [1, 1, 2, 2],
      data: [
        0,  1,  2,  3,
      ],
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [
        0, 0, 1, 0, 4, 6, 4, 12, 9,
      ],
    };
    testConvTranspose2d(input, filter, expected);
  });

  it('convTranspose2d options.groups', function() {
    const input = {
      shape: [1, 2, 2, 2],
      data: [
        2,  4,
        0,  1,
        2,  4,
        0,  1,
      ],
    };
    const filter = {
      shape: [2, 1, 2, 2],
      data: [
        3,  1,
        1,  5,
        3,  1,
        1,  5,
      ],
    };
    const options = {
      groups: 2,
    };
    const expected = {
      shape: [1, 2, 3, 3],
      data: [
        6, 14,  4,
        2, 17, 21,
        0,  1,  5,
        6, 14,  4,
        2, 17, 21,
        0,  1,  5,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.groups=2 options.strides=[2, 2]', function() {
    const input = {
      shape: [1, 2, 2, 2],
      data: [
        2,  4,
        0,  1,
        2,  4,
        0,  1,
      ],
    };
    const filter = {
      shape: [2, 1, 2, 2],
      data: [
        3,  1,
        1,  5,
        3,  1,
        1,  5,
      ],
    };
    const options = {
      groups: 2,
      strides: [2, 2],
    };
    const expected = {
      shape: [1, 2, 4, 4],
      data: [
        6,  2, 12,  4,
        2, 10,  4, 20,
        0,  0,  3,  1,
        0,  0,  1,  5,
        6,  2, 12,  4,
        2, 10,  4, 20,
        0,  0,  3,  1,
        0,  0,  1,  5,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.padding', function() {
    const input = {
      shape: [1, 1, 2, 2],
      data: [
        0,  1,  2,  3,
      ],
    };
    const filter = {
      shape: [1, 1, 2, 2],
      data: [
        0,  1,  2,  3,
      ],
    };
    const options = {
      padding: [1, 1, 1, 1],
    };
    const expected = {
      shape: [1, 1, 1, 1],
      data: [4],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.inputLayout=nchw', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Array(18).fill(1),
    };
    const options = {
      inputLayout: 'nchw',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.inputLayout=nhwc', function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Array(18).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
    };
    const expected = {
      shape: [1, 5, 5, 2],
      data: [
        0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,
        8.,  15., 15., 12., 12., 7.,  7.,  9.,  9.,  21., 21., 36., 36.,
        27., 27., 15., 15., 9.,  9.,  20., 20., 33., 33., 24., 24., 13.,
        13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.filterLayout=iohw', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Array(18).fill(1),
    };
    const options = {
      filterLayout: 'iohw',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.filterLayout=hwoi', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Array(18).fill(1),
    };
    const options = {
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.filterLayout=ohwi', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Array(18).fill(1),
    };
    const options = {
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.strides', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Array(18).fill(1),
    };
    const options = {
      strides: [3, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 2, 9, 7],
      data: [
        0., 0., 1.,  1., 3.,  2., 2.,
        0., 0., 1.,  1., 3.,  2., 2.,
        0., 0., 1.,  1., 3.,  2., 2.,
        3., 3., 7.,  4., 9.,  5., 5.,
        3., 3., 7.,  4., 9.,  5., 5.,
        3., 3., 7.,  4., 9.,  5., 5.,
        6., 6., 13., 7., 15., 8., 8.,
        6., 6., 13., 7., 15., 8., 8.,
        6., 6., 13., 7., 15., 8., 8.,
        0., 0., 1.,  1., 3.,  2., 2.,
        0., 0., 1.,  1., 3.,  2., 2.,
        0., 0., 1.,  1., 3.,  2., 2.,
        3., 3., 7.,  4., 9.,  5., 5.,
        3., 3., 7.,  4., 9.,  5., 5.,
        3., 3., 7.,  4., 9.,  5., 5.,
        6., 6., 13., 7., 15., 8., 8.,
        6., 6., 13., 7., 15., 8., 8.,
        6., 6., 13., 7., 15., 8., 8.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.outputSizes', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Array(18).fill(1),
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.outputPadding', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Array(18).fill(1),
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0.,
        0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d with padding', function() {
    const input = {
      shape: [1, 1, 2, 2],
      data: [
        0,  1,  2,  3,
      ],
    };
    const filter = {
      shape: [1, 1, 2, 2],
      data: [
        0,  1,  2,  3,
      ],
    };
    const options = {
      padding: [1, 1, 1, 1],
    };
    const expected = {
      shape: [1, 1, 1, 1],
      data: [4],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d options.dilations', function() {
    // this test is from ONNX
    //   https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/convtranspose.py#L328
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        3.0, 8.0, 1.0,
        9.0, 5.0, 7.0,
        3.0, 2.0, 6.0,
      ],
    };
    const filter = {
      shape: [2, 2, 1, 1],
      data: [
        7.0, 2.0,
        1.0, 9.0,
      ],
    };
    const options = {
      dilations: [2, 2],
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        21.0, 56.0, 13.0, 16.0, 2.0,
        63.0, 35.0, 67.0, 10.0, 14.0,
        24.0, 22.0, 76.0, 76.0, 21.0,
        9.0, 5.0, 88.0, 45.0, 63.0,
        3.0, 2.0, 33.0, 18.0, 54.0,
      ],
    };
    testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d bias', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [1],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        1, 2, 4, 4, 3, 4, 9, 16, 13, 8,
        10, 22, 37, 28, 16, 10, 21, 34, 25, 14,
        7, 14, 22, 16, 9,
      ],
    };
    testConvTranspose2d(input, filter, expected, options, bias);
  });

  it('convTranspose2d activation', function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        0, 1, 3, 3, 2, 3, 8, 15, 12, 7,
        9, 21, 36, 27, 15, 9, 20, 33, 24, 13,
        6, 13, 21, 15, 8,
      ],
    };
    testConvTranspose2d(input, filter, expected, options, undefined, 'relu');
  });
});
