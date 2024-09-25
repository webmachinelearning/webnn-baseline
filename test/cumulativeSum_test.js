'use strict';

import {cumulativeSum} from '../src/cumulativeSum.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test cumulativeSum', function() {
  function testCumulativeSum(input, axis, options={}, expected) {
    const tensor = new Tensor(input.shape, input.data);
    const outputTensor = cumulativeSum(tensor, axis, options);
    console.log('outputTensor', outputTensor);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.data);
  }

  it('test cumulativeSum 1d', function() {
    const input = {
      shape: [5],
      data: [
        1, 2, 3, 4, 5,
      ],
    };
    const axis=0;
    const options = {exclusive: 0, reverse: 0};
    const expected = {
      shape: [5],
      data: [
        1, 3, 6, 10, 15,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });

  it('test cumulativeSum 1d exclusive', function() {
    const input = {
      shape: [5],
      data: [
        1, 2, 3, 4, 5,
      ],
    };
    const axis=0;
    const options = {exclusive: 1, reverse: 0};
    const expected = {
      shape: [5],
      data: [
        0, 1, 3, 6, 10,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });

  it('test cumulativeSum 1d reverse', function() {
    const input = {
      shape: [5],
      data: [
        1, 2, 3, 4, 5,
      ],
    };
    const axis=0;
    const options = {exclusive: 0, reverse: 1};
    const expected = {
      shape: [5],
      data: [
        15, 14, 12, 9, 5,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });

  it('test cumulativeSum 1d reverse exclusive', function() {
    const input = {
      shape: [5],
      data: [
        1, 2, 3, 4, 5,
      ],
    };
    const axis=0;
    const options = {exclusive: 1, reverse: 1};
    const expected = {
      shape: [5],
      data: [
        14, 12, 9, 5, 0,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });

  it('test cumulativeSum 2d', function() {
    const input = {
      shape: [2, 3],
      data: [
        1, 2, 3, 4, 5, 6,
      ],
    };
    const axis=0;
    const options = {exclusive: 0, reverse: 0};
    const expected = {
      shape: [2, 3],
      data: [
        1, 2, 3, 5, 7, 9,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });

  it('test cumulativeSum 2d axis=1', function() {
    const input = {
      shape: [2, 3],
      data: [
        1, 2, 3, 4, 5, 6,
      ],
    };
    const axis=1;
    const options = {exclusive: 0, reverse: 0};
    const expected = {
      shape: [2, 3],
      data: [
        1, 3, 6, 4, 9, 15,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });

  it('test cumulativeSum 2d negtive axis', function() {
    const input = {
      shape: [2, 3],
      data: [
        1, 2, 3, 4, 5, 6,
      ],
    };
    const axis=1;
    const options = {exclusive: 0, reverse: 0};
    const expected = {
      shape: [2, 3],
      data: [
        1, 3, 6, 4, 9, 15,
      ],
    };
    testCumulativeSum(input, axis, options, expected);
  });
});


