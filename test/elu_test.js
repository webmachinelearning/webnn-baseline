'use strict';

import {elu} from '../src/elu.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test elu', function() {
  function testElu(inputShape, inputValue, expected, options = {}) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = elu(inputTensor, options);
    utils.checkValue(outputTensor, expected);
  }

  it('elu default', function() {
    testElu([3], [-1, 0, 1], [-0.6321205588285577, 0, 1]);
  });

  it('elu', function() {
    testElu([3], [-1, 0, 1], [-1.2642411176571153, 0, 1], {alpha: 2});
    testElu([1, 1, 1, 3], [-1, 0, 1], [1.2642411176571153, 0, 1], {alpha: -2});
  });
});
