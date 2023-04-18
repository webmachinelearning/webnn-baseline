'use strict';

import {instanceNormalization} from '../src/instance_normalization.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test instanceNormalization', function() {
  function testInstanceNorm(
      input, expected, scale = undefined, bias = undefined, options = {}) {
    const inputTensor = new Tensor(input.shape, input.value);
    if (scale) {
      options.scale = new Tensor(scale.shape, scale.value);
    }
    if (bias) {
      options.bias = new Tensor(bias.shape, bias.value);
    }
    const outputTensor = instanceNormalization(inputTensor, options);
    utils.checkShape(outputTensor, input.shape);
    utils.checkValue(outputTensor, expected);
  }

  it('instanceNormalization default', function() {
    testInstanceNorm(
        {
          shape: [1, 2, 1, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [
          -1.2247356859083902,
          0,
          1.2247356859083902,
          -1.2247356859083902,
          0,
          1.2247356859083902,
        ],
    );
  });

  it('instanceNormalization with scale', function() {
    testInstanceNorm(
        {
          shape: [1, 2, 1, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [
          -1.2247356859083902,
          0,
          1.2247356859083902,
          -1.8371035288625852,
          0,
          1.8371035288625852,
        ],
        {
          shape: [2],
          value: [1.0, 1.5],
        },
    );
  });

  it('instanceNormalization with bias', function() {
    testInstanceNorm(
        {
          shape: [1, 2, 1, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [
          -1.2247356859083902,
          0,
          1.2247356859083902,
          -0.22473568590839021,
          1,
          2.2247356859083904,
        ],
        undefined,
        {
          shape: [2],
          value: [0, 1],
        },
    );
  });

  it('instanceNormalization with epsilon', function() {
    testInstanceNorm(
        {
          shape: [1, 2, 1, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [
          -1.2247333895698826,
          0,
          1.2247333895698826,
          -1.2247333895698826,
          0,
          1.2247333895698826,
        ],
        undefined,
        undefined,
        {
          epsilon: 0.0000125,
        },
    );
  });

  it('instanceNormalization with nchw layout', function() {
    testInstanceNorm(
        {
          shape: [1, 2, 1, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [
          -1.2247356859083902,
          0,
          1.2247356859083902,
          -1.2247356859083902,
          0,
          1.2247356859083902,
        ],
        undefined,
        undefined,
        {
          layout: 'nchw',
        },
    );
  });

  it('instanceNormalization with nhwc layout', function() {
    testInstanceNorm(
        {
          shape: [1, 1, 3, 2],
          value: [-1, 2, 0, 3, 1, 4],
        },
        [
          -1.2247356859083902,
          -1.2247356859083902,
          0,
          0,
          1.2247356859083902,
          1.2247356859083902,
        ],
        undefined,
        undefined,
        {
          layout: 'nhwc',
        },
    );
  });

  it('instanceNormalization all options', function() {
    testInstanceNorm(
        {
          shape: [3, 1, 1, 2],
          value: [
            -1, 2, 0, 3, 1, 4,
          ],
        },
        [
          0, 1, 0, 1, 0, 1,
        ],
        {
          shape: [2],
          value: [1.0, 1.5],
        },
        {
          shape: [2],
          value: [0, 1],
        },
        {
          epsilon: 0.0000125,
          layout: 'nhwc',
        },
    );
  });
});
