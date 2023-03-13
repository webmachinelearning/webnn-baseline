'use strict';

import {hardSwish} from '../src/hard_swish.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test hardSwish', function() {
  function testHardSwish(input, expected) {
    const inputTensor = new Tensor(input.shape, input.value);
    const outputTensor = hardSwish(inputTensor);
    utils.checkValue(outputTensor, expected);
  }

  // this input data referres to NNAPI CTS tests of HARD_SWISH op
  //   https://android.googlesource.com/platform/frameworks/ml/+ \
  //       /master/nn/runtime/test/specs/V1_3/hard_swish.mod.py#50
  const inputData = [
    4.53125, 3.90625, 3.046875, -8.59375, -1.328125,
    1.328125, 0.0, -8.515625, -8.984375, -0.234375,
    0.859375, 9.84375, -0.15625, -8.515625, 8.671875,
    4.609375, 9.21875, -1.796875, 1.171875, 9.375,
    -8.75, 2.421875, -8.125, -1.09375, -9.609375,
    -1.015625, -9.84375, 2.578125, 4.921875, -5.078125,
    5.0, -0.859375, 1.953125, -6.640625, -7.8125,
    4.453125, -4.453125, -6.875, 0.78125, 0.859375,
  ];
  const expectedData = [
    4.53125, 3.90625, 3.046875, -0.0, -0.3700764973958333,
    0.9580485026041666, 0.0, -0.0, -0.0, -0.1080322265625,
    0.5527750651041666, 9.84375, -0.07405598958333333, -0.0, 8.671875,
    4.609375, 9.21875, -0.3603108723958333, 0.8148193359375, 9.375,
    -0.0, 2.1885172526041665, -0.0, -0.3474934895833333, -0.0,
    -0.3358968098958333, -0.0, 2.3968505859375, 4.921875, -0.0,
    5.0, -0.3065999348958333, 1.6123453776041667, -0.0, -0.0,
    4.453125, -0.0, -0.0, 0.4923502604166667, 0.5527750651041666,
  ];

  it('hardSwish 1D', function() {
    testHardSwish({shape: [40], value: inputData}, expectedData);
  });

  it('hardSwish 5D', function() {
    testHardSwish({shape: [1, 2, 2, 2, 5], value: inputData}, expectedData);
  });
});
