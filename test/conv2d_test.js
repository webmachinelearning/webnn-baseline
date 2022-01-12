'use strict';

import {conv2d} from '../src/conv2d.js';
import {clamp} from '../src/clamp.js';
import {leakyRelu} from '../src/leaky_relu.js';
import {relu} from '../src/relu.js';
import {sigmoid} from '../src/sigmoid.js';
import {Tensor} from '../src/tensor.js';

import * as utils from './utils.js';

describe('test conv2d', function() {
  function testConv2d(
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

    const outputTensor = conv2d(inputTensor, filterTensor, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('conv2d with padding default', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const options = {padding: [1, 1, 1, 1]};
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d input channels with padding default', function() {
    const input = {
      shape: [1, 3, 2, 3],
      data: [
        1.0, 4.0, 7.0, 10.0, 13.0, 16.0,
        2.0, 5.0, 8.0, 11.0, 14.0, 17.0,
        3.0, 6.0, 9.0, 12.0, 15.0, 18.0,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
    };
    const options = {};
    const expected = {
      shape: [1, 3, 2, 3],
      data: [
        30.0, 66.0, 102.0, 138.0, 174.0, 210.0,
        36.0, 81.0, 126.0, 171.0, 216.0, 261.0,
        42.0, 96.0, 150.0, 204.0, 258.0, 312.0,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding explicit autoPad', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      autoPad: 'explicit',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw oihw', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw hwio', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw ohwi', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw ihwo', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc oihw', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc hwio', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc ohwi', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc ihwo', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding default', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    testConv2d(input, filter, expected);
  });

  it('conv2d without padding nchw hwio', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nchw ohwi', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nchw ihwo', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc oihw', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc hwio', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc ohwi', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc ihwo', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding default', function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nchw hwio', function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nchw ohwi', function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nchw ihwo', function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc oihw', function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc hwio', function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc ohwi', function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc ihwo', function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding default', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 4, 2],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nchw hwio', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [4, 2, 1, 1],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nchw ohwi', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nchw ihwo', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc oihw', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 4, 2],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc hwio', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [4, 2, 1, 1],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc ohwi', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc ihwo', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower default', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nchw hwio', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nchw ohwi', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nchw ihwo', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc oihw', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc hwio', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc ohwi', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc ihwo', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper default', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nchw hwio', function() {
    const input = {
      shape: [1, 1, 4, 4],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 2, 2],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nchw ohwi', function() {
    const input = {
      shape: [1, 1, 4, 4],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 2, 2],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nchw ihwo', function() {
    const input = {
      shape: [1, 1, 4, 4],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 2, 2],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc oihw', function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc hwio', function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc ohwi', function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc ihwo', function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });

  it('fused depthwise conv2d default', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: [
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        10,
        20,
        30,
        40,
        0,
        0,
        0,
        0,
      ],
    };
    const filter = {
      shape: [4, 1, 2, 2],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        50.0,
        50.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {groups: 4};
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 4, 1, 1],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nchw hwio', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: [
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        10,
        20,
        30,
        40,
        0,
        0,
        0,
        0,
      ],
    };
    const filter = {
      shape: [2, 2, 1, 4],
      data: [
        0.25,
        0.0,
        10.0,
        50.0,
        0.25,
        1.0,
        20.0,
        50.0,
        0.25,
        0.0,
        30.0,
        50.0,
        0.25,
        1.0,
        40.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 4, 1, 1],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nchw ohwi', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: [
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        10,
        20,
        30,
        40,
        0,
        0,
        0,
        0,
      ],
    };
    const filter = {
      shape: [4, 2, 2, 1],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        50.0,
        50.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 4, 1, 1],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nchw ihwo', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: [
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        10,
        20,
        30,
        40,
        0,
        0,
        0,
        0,
      ],
    };
    const filter = {
      shape: [1, 2, 2, 4],
      data: [
        0.25,
        0.0,
        10.0,
        50.0,
        0.25,
        1.0,
        20.0,
        50.0,
        0.25,
        0.0,
        30.0,
        50.0,
        0.25,
        1.0,
        40.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 4, 1, 1],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nhwc oihw', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: [
        10,
        21,
        10,
        0,
        10,
        22,
        20,
        0,
        10,
        23,
        30,
        0,
        10,
        24,
        40,
        0,
      ],
    };
    const filter = {
      shape: [4, 1, 2, 2],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        50.0,
        50.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 1, 4],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nhwc hwio', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: [
        10,
        21,
        10,
        0,
        10,
        22,
        20,
        0,
        10,
        23,
        30,
        0,
        10,
        24,
        40,
        0,
      ],
    };
    const filter = {
      shape: [2, 2, 1, 4],
      data: [
        0.25,
        0.0,
        10.0,
        50.0,
        0.25,
        1.0,
        20.0,
        50.0,
        0.25,
        0.0,
        30.0,
        50.0,
        0.25,
        1.0,
        40.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 1, 4],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nhwc ohwi', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: [
        10,
        21,
        10,
        0,
        10,
        22,
        20,
        0,
        10,
        23,
        30,
        0,
        10,
        24,
        40,
        0,
      ],
    };
    const filter = {
      shape: [4, 2, 2, 1],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        50.0,
        50.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 1, 4],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused depthwise conv2d nhwc ihwo', function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: [
        10,
        21,
        10,
        0,
        10,
        22,
        20,
        0,
        10,
        23,
        30,
        0,
        10,
        24,
        40,
        0,
      ],
    };
    const filter = {
      shape: [1, 2, 2, 4],
      data: [
        0.25,
        0.0,
        10.0,
        50.0,
        0.25,
        1.0,
        20.0,
        50.0,
        0.25,
        0.0,
        30.0,
        50.0,
        0.25,
        1.0,
        40.0,
        50.0,
      ],
    };
    const bias = {
      shape: [4],
      data: [6000, 7000, 8000, 9000],
    };
    let expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options, bias);
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 1, 4],
      data: [6, 6, 6, 6],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('depthwise conv2d nchw oihw', function() {
    const input = {
      shape: [1, 4, 2, 2],
      data: [
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        10,
        20,
        30,
        40,
        0,
        0,
        0,
        0,
      ],
    };
    const filter = {
      shape: [4, 1, 2, 2],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        50.0,
        50.0,
        50.0,
      ],
    };
    let expected = {
      shape: [1, 4, 1, 1],
      data: [10, 46, 3000, 0],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'oihw',
    };
    testConv2d(input, filter, expected, options);
    testConv2d(input, filter, expected, options, undefined, 'relu', true);
    expected = {
      shape: [1, 4, 1, 1],
      data: [6, 6, 6, 0],
    };
    testConv2d(input, filter, expected, options, undefined, 'relu6', true);
  });

  it('fused depthwise conv2d explicit autoPad', function() {
    const input = {
      shape: [1, 2, 3, 3],
      data: [
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
      ],
    };
    const filter = {
      shape: [2, 1, 2, 2],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
      ],
    };
    let expected = {
      shape: [1, 2, 3, 3],
      data: [
        10,
        10,
        5,
        10,
        10,
        5,
        5,
        5,
        2.5,
        47,
        49,
        0,
        53,
        55,
        0,
        28,
        29,
        0,
      ],
    };
    const options = {
      groups: 2,
      padding: [0, 1, 0, 1],
      autoPad: 'explicit',
    };
    testConv2d(input, filter, expected, options);
    testConv2d(input, filter, expected, options, undefined, 'relu', true);
    expected = {
      shape: [1, 2, 3, 3],
      data: [
        6,
        6,
        5,
        6,
        6,
        5,
        5,
        5,
        2.5,
        6,
        6,
        0,
        6,
        6,
        0,
        6,
        6,
        0,
      ],
    };
    testConv2d(input, filter, expected, options, undefined, 'relu6', true);
  });

  it('fused depthwise conv2d same-upper autoPad', function() {
    const input = {
      shape: [1, 2, 3, 3],
      data: [
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
      ],
    };
    const filter = {
      shape: [2, 1, 2, 2],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
      ],
    };
    let expected = {
      shape: [1, 2, 3, 3],
      data: [
        10,
        10,
        5,
        10,
        10,
        5,
        5,
        5,
        2.5,
        47,
        49,
        0,
        53,
        55,
        0,
        28,
        29,
        0,
      ],
    };
    const options = {
      groups: 2,
      autoPad: 'same-upper',
    };
    testConv2d(input, filter, expected, options);
    testConv2d(input, filter, expected, options, undefined, 'relu', true);
    expected = {
      shape: [1, 2, 3, 3],
      data: [
        6,
        6,
        5,
        6,
        6,
        5,
        5,
        5,
        2.5,
        6,
        6,
        0,
        6,
        6,
        0,
        6,
        6,
        0,
      ],
    };
    testConv2d(input, filter, expected, options, undefined, 'relu6', true);
  });

  it('fused depthwise conv2d same-lower autoPad', function() {
    const input = {
      shape: [1, 2, 3, 3],
      data: [
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
      ],
    };
    const filter = {
      shape: [2, 1, 2, 2],
      data: [
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        1.0,
        0.0,
        1.0,
      ],
    };
    let expected = {
      shape: [1, 2, 3, 3],
      data: [
        2.5,
        5,
        5,
        5,
        10,
        10,
        5,
        10,
        10,
        21,
        22,
        23,
        45,
        47,
        49,
        51,
        53,
        55,
      ],
    };
    const options = {
      groups: 2,
      autoPad: 'same-lower',
    };
    testConv2d(input, filter, expected, options);
    testConv2d(input, filter, expected, options, undefined, 'relu', true);
    expected = {
      shape: [1, 2, 3, 3],
      data: [
        2.5,
        5,
        5,
        5,
        6,
        6,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
      ],
    };
    testConv2d(input, filter, expected, options, undefined, 'relu6', true);
  });

  it('fused conv2d with padding default', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
    };
    let expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 5, 5],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
    expected = {
      shape: [1, 1, 5, 5],
      data: [
        -8.800000131130219,
        -7.900000117719173,
        -7.300000108778477,
        -6.70000009983778,
        -7.600000113248825,
        -6.70000009983778,
        -4.6000000685453415,
        -3.7000000551342964,
        -2.8000000417232513,
        -4.90000007301569,
        -3.7000000551342964,
        -0.10000000149011612,
        8,
        17,
        -1.9000000283122063,
        -0.7000000104308128,
        44,
        53,
        62,
        11,
        -2.8000000417232513,
        11,
        17,
        23,
        -1.600000023841858,
      ],
    };
    testConv2d(
        input, filter, expected, options, bias, 'leakyRelu', true,
        {alpha: 0.10000000149011612});
    expected = {
      shape: [1, 1, 5, 5],
      data: [
        6.054601895401186e-39,
        4.906094730649281e-35,
        1.9792598779469048e-32,
        7.984904245686979e-30,
        9.854154686111257e-34,
        7.984904245686979e-30,
        1.0530617357553813e-20,
        8.533047625744066e-17,
        6.914400106935423e-13,
        5.242885663363465e-22,
        8.533047625744066e-17,
        0.2689414213699951,
        0.9996646498695336,
        0.9999999586006244,
        5.602796406145939e-9,
        0.0009110511944006454,
        1,
        1,
        1,
        0.999983298578152,
        6.914400106935423e-13,
        0.999983298578152,
        0.9999999586006244,
        0.9999999998973812,
        1.12535162055095e-7,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'sigmoid', true);
  });

  it('fused conv2d with padding nchw hwio', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    let expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 5, 5],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused conv2d with padding nchw ohwi', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    let expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 5, 5],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused conv2d with padding nchw ihwo', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    let expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 1, 5, 5],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused conv2d with padding nhwc oihw', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    let expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 5, 5, 1],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
    expected = {
      shape: [1, 5, 5, 1],
      data: [
        -8.800000131130219,
        -7.900000117719173,
        -7.300000108778477,
        -6.70000009983778,
        -7.600000113248825,
        -6.70000009983778,
        -4.6000000685453415,
        -3.7000000551342964,
        -2.8000000417232513,
        -4.90000007301569,
        -3.7000000551342964,
        -0.10000000149011612,
        8,
        17,
        -1.9000000283122063,
        -0.7000000104308128,
        44,
        53,
        62,
        11,
        -2.8000000417232513,
        11,
        17,
        23,
        -1.600000023841858,
      ],
    };
    testConv2d(
        input, filter, expected, options, bias, 'leakyRelu', true,
        {alpha: 0.10000000149011612});
    expected = {
      shape: [1, 5, 5, 1],
      data: [
        6.054601895401186e-39,
        4.906094730649281e-35,
        1.9792598779469048e-32,
        7.984904245686979e-30,
        9.854154686111257e-34,
        7.984904245686979e-30,
        1.0530617357553813e-20,
        8.533047625744066e-17,
        6.914400106935423e-13,
        5.242885663363465e-22,
        8.533047625744066e-17,
        0.2689414213699951,
        0.9996646498695336,
        0.9999999586006244,
        5.602796406145939e-9,
        0.0009110511944006454,
        1,
        1,
        1,
        0.999983298578152,
        6.914400106935423e-13,
        0.999983298578152,
        0.9999999586006244,
        0.9999999998973812,
        1.12535162055095e-7,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'sigmoid', true);
  });

  it('fused conv2d with padding nhwc hwio', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    let expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 5, 5, 1],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused conv2d with padding nhwc ohwi', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    let expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 5, 5, 1],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('fused conv2d with padding nhwc ihwo', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: [-100],
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    let expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu');
    testConv2d(input, filter, expected, options, bias, 'relu', true);
    expected = {
      shape: [1, 5, 5, 1],
      data: [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
        6., 0., 0., 6., 6., 6., 6., 0., 6., 6., 6., 0.,
      ],
    };
    testConv2d(input, filter, expected, options, bias, 'relu6', true);
  });

  it('conv2d input=1x1x5x5 dilations=2', function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 1, 1],
      data: [108],
    };
    const options = {
      dilations: [2, 2],
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d input=1x5x5x1 dilations=4 nhwc', function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ],
    };
    const filter = {
      shape: [1, 1, 2, 2],
      data: new Array(4).fill(1),
    };
    const expected = {
      shape: [1, 1, 1, 1],
      data: [48],
    };
    const options = {
      dilations: [4, 4],
      inputLayout: 'nhwc',
    };
    testConv2d(input, filter, expected, options);
  });

  it('conv2d input=1x65x65x1 dilations=4 nhwc', function() {
    const input = {
      shape: [1, 65, 65, 1],
      data: [
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,
        2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,
        5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,
        2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,
        5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,
        2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,
        5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 57, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
      ],
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
      ],
    };
    const expected = {
      shape: [1, 57, 57, 1],
      data: [
        15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,
        57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,
        99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138,
        151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180,
        183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,
        54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,
        96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135,
        138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177,
        180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,
        51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,
        93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132,
        135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174,
        177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,
        48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,
        90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139,
        132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171,
        174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,
        45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,
        87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126,
        139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168,
        171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,
        42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,
        84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123,
        126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165,
        168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,
        39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,
        81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120,
        123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162,
        165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,
        36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,
        78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117,
        120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159,
        162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,
        33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,
        75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114,
        117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156,
        159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,
        30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,
        72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111,
        114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163,
        156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,
        27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,
        69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108,
        111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150,
        163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,
        24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,
        66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105,
        108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147,
        150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,
        21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,
        63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102,
        105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144,
        147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,
        18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,
        60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,
        102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151,
        144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183,
        15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,
        57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,
        99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138,
        151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180,
        183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,
        54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,
        96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135,
        138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177,
        180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,
        51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,
        93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132,
        135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174,
        177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,
        48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,
        90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139,
        132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171,
        174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,
        45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,
        87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126,
        139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168,
        171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,
        42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,
        84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123,
        126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165,
        168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,
        39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,
        81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120,
        123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162,
        165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,
        36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,
        78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117,
        120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159,
        162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,
        33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,
        75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114,
        117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156,
        159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,
        30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,
        72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111,
        114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163,
        156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,
        27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,
        69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108,
        111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150,
        163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,
        24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,
        66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105,
        108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147,
        150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,
        21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,
        63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102,
        105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144,
        147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,
        18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,
        60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,
        102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151,
        144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183,
        15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,
        57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,
        99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138,
        151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180,
        183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,
        54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,
        96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135,
        138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177,
        180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,
        51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,
        93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132,
        135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174,
        177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,
        48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,
        90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139,
        132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171,
        174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,
        45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,
        87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126,
        139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168,
        171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,
        42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,
        84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123,
        126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165,
        168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,
        39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,
        81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120,
        123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162,
        165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,
        36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,
        78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117,
        120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159,
        162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,
        33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,
        75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114,
        117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156,
        159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,
        30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,
        72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111,
        114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163,
        156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,
        27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,
        69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108,
        111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150,
        163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,
        24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,
        66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105,
        108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147,
        150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,
        21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,
        63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102,
        105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144,
        147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,
        18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,
        60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,
        102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151,
        144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183,
        15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,
        57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,
        99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138,
        151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180,
        183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,
        54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,
        96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135,
        138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177,
        180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,
        51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,
        93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132,
        135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174,
        177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,
        48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,
        90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139,
        132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171,
        174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,  42,
        45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,
        87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123, 126,
        139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165, 168,
        171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,  39,
        42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,
        84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120, 123,
        126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162, 165,
        168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,  36,
        39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,
        81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117, 120,
        123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159, 162,
        165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,  33,
        36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,  75,
        78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114, 117,
        120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156, 159,
        162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,  30,
        33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,
        75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111, 114,
        117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163, 156,
        159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,  27,
        30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,
        72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108, 111,
        114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150, 163,
        156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,  24,
        27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,
        69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105, 108,
        111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147, 150,
        163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,  21,
        24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,
        66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102, 105,
        108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144, 147,
        150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,  18,
        21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,
        63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,  102,
        105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151, 144,
        147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 15,
        18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,
        60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,  99,
        102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138, 151,
        144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183,
        15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,  48,  51,  54,
        57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,  96,
        99,  102, 105, 108, 111, 114, 117, 120, 123, 126, 139, 132, 135, 138,
        151, 144, 147, 150, 163, 156, 159, 162, 165, 168, 171, 174, 177, 180,
        183,
      ],
    };
    const options = {
      dilations: [4, 4],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    testConv2d(input, filter, expected, options);
  });
});
