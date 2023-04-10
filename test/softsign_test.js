'use strict';

import {softsign} from '../src/softsign.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test softsign', function() {
  function testSoftsign(input, expected, shape) {
    const x = new Tensor(shape, input);
    const y = softsign(x);
    utils.checkShape(y, shape);
    utils.checkValue(y, expected);
  }

  it('softsign', function() {
    testSoftsign([-1, 0, 1], [-0.5, 0, 0.5], [3]);
  });
});
