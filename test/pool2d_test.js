'use strict';
import {averagePool2d, maxPool2d} from '../src/pool2d.js';
import {Tensor} from '../src/lib/tensor.js';

import * as utils from './utils.js';

describe('test pool2d', function() {
  it('maxPool2d default', function() {
    const x = new Tensor([1, 1, 4, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const y = maxPool2d(x, {windowDimensions});
    utils.checkShape(y, [1, 1, 2, 2]);
    utils.checkValue(y, [11, 12, 15, 16]);
  });

  it('maxPool2d nhwc', function() {
    const x = new Tensor([1, 4, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const layout = 'nhwc';
    const y = maxPool2d(x, {windowDimensions, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    utils.checkValue(y, [11, 12, 15, 16]);
  });

  it('maxPool2d dilations default', function() {
    const x = new Tensor([1, 1, 4, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [2, 2];
    const dilations = [2, 2];
    const y = maxPool2d(x, {windowDimensions, dilations});
    utils.checkShape(y, [1, 1, 2, 2]);
    utils.checkValue(y, [11, 12, 15, 16]);
  });

  it('maxPool2d dilations nhwc', function() {
    const x = new Tensor([1, 4, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [2, 2];
    const dilations = [2, 2];
    const layout = 'nhwc';
    const y = maxPool2d(x, {windowDimensions, dilations, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    utils.checkValue(y, [11, 12, 15, 16]);
  });

  it('maxPool2d pads default', function() {
    const x = new Tensor([1, 1, 5, 5], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const y = maxPool2d(x, {windowDimensions, padding});
    utils.checkShape(y, [1, 1, 5, 5]);
    const expected = [
      13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
      25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25,
    ];
    utils.checkValue(y, expected);
  });

  it('maxPool2d pads nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const layout = 'nhwc';
    const y = maxPool2d(x, {windowDimensions, padding, layout});
    utils.checkShape(y, [1, 5, 5, 1]);
    const expected = [
      13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
      25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25,
    ];
    utils.checkValue(y, expected);
  });


  it('maxPool2d pad nhwc', function() {
    const x = new Tensor([1, 7, 7, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const padding = [2, 1, 2, 1];
    const strides = [2, 2];
    const layout = 'nhwc';
    const y = maxPool2d(
        x, {windowDimensions, padding, strides, layout});
    utils.checkShape(y, [1, 4, 4, 1]);
    const expected = [
      9,
      11,
      13,
      14,
      23,
      25,
      27,
      28,
      37,
      39,
      41,
      42,
      44,
      46,
      48,
      49,
    ];
    utils.checkValue(y, expected);
  });

  it('maxPool2d strides default', function() {
    const x = new Tensor([1, 1, 5, 5], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const y = maxPool2d(x, {windowDimensions, strides});
    utils.checkShape(y, [1, 1, 2, 2]);
    const expected = [7, 9, 17, 19];
    utils.checkValue(y, expected);
  });

  it('maxPool2d strides nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const layout = 'nhwc';
    const y = maxPool2d(x, {windowDimensions, strides, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    const expected = [7, 9, 17, 19];
    utils.checkValue(y, expected);
  });

  it('averagePool2d default', function() {
    const x = new Tensor([1, 1, 4, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const y = averagePool2d(x, {windowDimensions});
    utils.checkShape(y, [1, 1, 2, 2]);
    const expected = [6, 7, 10, 11];
    utils.checkValue(y, expected);
  });

  it('averagePool2d nhwc', function() {
    const x = new Tensor([1, 4, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const layout = 'nhwc';
    const y = averagePool2d(x, {windowDimensions, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    const expected = [6, 7, 10, 11];
    utils.checkValue(y, expected);
  });

  it('averagePool2d pads default', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const layout = 'nhwc';
    const y = averagePool2d(x, {windowDimensions, padding, layout});
    utils.checkShape(y, [1, 5, 5, 1]);
    const expected = [
      7,    7.5, 8,    8.5, 9,    9.5, 10,   10.5, 11,   11.5, 12,   12.5, 13,
      13.5, 14,  14.5, 15,  15.5, 16,  16.5, 17,   17.5, 18,   18.5, 19,
    ];
    utils.checkValue(y, expected);
  });

  it('averagePool2d pads nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const layout = 'nhwc';
    const y = averagePool2d(x, {windowDimensions, padding, layout});
    utils.checkShape(y, [1, 5, 5, 1]);
    const expected = [
      7,    7.5, 8,    8.5, 9,    9.5, 10,   10.5, 11,   11.5, 12,   12.5, 13,
      13.5, 14,  14.5, 15,  15.5, 16,  16.5, 17,   17.5, 18,   18.5, 19,
    ];
    utils.checkValue(y, expected);
  });


  it('averagePool2d Pad nhwc', function() {
    const x = new Tensor([1, 7, 7, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const padding = [2, 1, 2, 1];
    const strides = [2, 2];
    const layout = 'nhwc';
    const y = averagePool2d(
        x, {windowDimensions, padding, strides, layout});
    utils.checkShape(y, [1, 4, 4, 1]);
    const expected = [
      5,
      6,
      8,
      9.5,
      12,
      13,
      15,
      16.5,
      26,
      27,
      29,
      30.5,
      36.5,
      37.5,
      39.5,
      41,
    ];
    utils.checkValue(y, expected);
  });

  it('averagePool2d strides default', function() {
    const x = new Tensor([1, 1, 5, 5], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const y = averagePool2d(x, {windowDimensions, strides});
    utils.checkShape(y, [1, 1, 2, 2]);
    const expected = [4, 6, 14, 16];
    utils.checkValue(y, expected);
  });

  it('averagePool2d strides nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const layout = 'nhwc';
    const y = averagePool2d(x, {windowDimensions, strides, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    const expected = [4, 6, 14, 16];
    utils.checkValue(y, expected);
  });

  it('averagePool2d pads outputSizes=[3,3]', function() {
    const x = new Tensor([1, 1, 7, 7], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const strides = [2, 2];
    const padding = [1, 1, 1, 1];
    const outputSizes = [3, 3];
    const y =
        averagePool2d(x, {windowDimensions, strides, padding, outputSizes});
    utils.checkShape(y, [1, 1, 3, 3]);
    const expected = [9, 10.5, 12.5, 19.5, 21, 23, 33.5, 35, 37];
    utils.checkValue(y, expected);
  });

  it('averagePool2d pads outputSizes=[4,4]', function() {
    const x = new Tensor([1, 1, 7, 7], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const strides = [2, 2];
    const padding = [1, 1, 1, 1];
    const outputSizes = [4, 4];
    const y =
        averagePool2d(x, {windowDimensions, strides, padding, outputSizes});
    utils.checkShape(y, [1, 1, 4, 4]);
    const expected = [9, 10.5, 12.5, 13.5, 19.5, 21, 23, 24, 33.5, 35, 37, 38, 40.5, 42, 44, 45];
    utils.checkValue(y, expected);
  });

  it('averagePool2d pads roundingType=floor', function() {
    const x = new Tensor([1, 1, 7, 7], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const strides = [2, 2];
    const padding = [1, 1, 1, 1];
    const roundingType = 'floor';
    const y =
        averagePool2d(x, {windowDimensions, strides, padding, roundingType});
    utils.checkShape(y, [1, 1, 3, 3]);
    const expected = [9, 10.5, 12.5, 19.5, 21, 23, 33.5, 35, 37];
    utils.checkValue(y, expected);
  });

  it('averagePool2d pads roundingType=ceil', function() {
    const x = new Tensor([1, 1, 7, 7], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const strides = [2, 2];
    const padding = [1, 1, 1, 1];
    const roundingType = 'ceil';
    const y =
        averagePool2d(x, {windowDimensions, strides, padding, roundingType});
    utils.checkShape(y, [1, 1, 4, 4]);
    const expected = [9, 10.5, 12.5, 13.5, 19.5, 21, 23, 24, 33.5, 35, 37, 38, 40.5, 42, 44, 45];
    utils.checkValue(y, expected);
  });

  it('global averagePool2d default', function() {
    const x = new Tensor([1, 3, 5, 5], [
      -1.1289884,  0.34016284,  0.497431,    2.1915932,   0.42038894,
      -0.18261199, -0.15769927, -0.26465914, 0.03877424,  0.39492005,
      -0.33410737, 0.74918455,  -1.3542547,  -0.0222946,  0.7094626,
      -0.09399617, 0.790736,    -0.75826526, 0.27656242,  0.46543223,
      -1.2342638,  1.1549494,   0.24823844,  0.75670505,  -1.7108902,
      -1.4767597,  -1.4969662,  -0.31936142, 0.5327554,   -0.06070877,
      0.31212643,  2.2274113,   1.2775147,   0.59886885,  -1.5765078,
      0.18522178,  0.22655599,  0.88869494,  0.38609484,  -0.05860576,
      -0.72732115, -0.0046324,  -1.3593693,  -0.6295078,  1.384531,
      0.06825881,  0.19907428,  0.20298219,  -0.8399954,  1.3583295,
      0.02117888,  -1.0636739,  -0.30460566, -0.92678875, -0.09120782,
      -0.88333017, -0.9641269,  0.6065926,   -0.5830042,  -0.81138134,
      1.3569402,   1.2891295,   0.2508177,   0.20211531,  0.8832168,
      -0.19886094, -0.61088,    0.682026,    -0.5253442,  1.5022339,
      1.0256356,   1.0642492,   -0.4169051,  -0.8740329,  1.1494869,
    ]);
    const y = averagePool2d(x);
    utils.checkShape(y, [1, 3, 1, 1]);
    const expected = [
      0.07170040239999997,
      0.05194737240000002,
      0.07117922839999995,
    ];
    utils.checkValue(y, expected);
  });

  it('global averagePool2d nhwc', function() {
    const x = new Tensor([1, 5, 5, 3], [
      -1.1289884,  -1.4767597,  0.02117888,  0.34016284,  -1.4969662,
      -1.0636739,  0.497431,    -0.31936142, -0.30460566, 2.1915932,
      0.5327554,   -0.92678875, 0.42038894,  -0.06070877, -0.09120782,
      -0.18261199, 0.31212643,  -0.88333017, -0.15769927, 2.2274113,
      -0.9641269,  -0.26465914, 1.2775147,   0.6065926,   0.03877424,
      0.59886885,  -0.5830042,  0.39492005,  -1.5765078,  -0.81138134,
      -0.33410737, 0.18522178,  1.3569402,   0.74918455,  0.22655599,
      1.2891295,   -1.3542547,  0.88869494,  0.2508177,   -0.0222946,
      0.38609484,  0.20211531,  0.7094626,   -0.05860576, 0.8832168,
      -0.09399617, -0.72732115, -0.19886094, 0.790736,    -0.0046324,
      -0.61088,    -0.75826526, -1.3593693,  0.682026,    0.27656242,
      -0.6295078,  -0.5253442,  0.46543223,  1.384531,    1.5022339,
      -1.2342638,  0.06825881,  1.0256356,   1.1549494,   0.19907428,
      1.0642492,   0.24823844,  0.20298219,  -0.4169051,  0.75670505,
      -0.8399954,  -0.8740329,  -1.7108902,  1.3583295,   1.1494869,
    ]);
    const layout = 'nhwc';
    const y = averagePool2d(x, {layout});
    utils.checkShape(y, [1, 1, 1, 3]);
    const expected = [
      0.07170040239999997,
      0.05194737240000002,
      0.07117922839999995,
    ];
    utils.checkValue(y, expected);
  });
});
