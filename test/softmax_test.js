'use strict';

import {softmax} from '../src/softmax.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test softmax', function() {
  it('softmax', function() {
    const x = new Tensor([3, 4], [
      0.4301911,
      0.54719144,
      -1.1637765,
      0.18390046,
      0.58390397,
      0.1735679,
      0.539724,
      -0.953514,
      -0.59202826,
      -0.17344485,
      0.14395015,
      -0.37920907,
    ]);
    const y = softmax(x);
    utils.checkShape(y, [3, 4]);
    const expected = [
      0.3216537706230936,
      0.36157737612692964,
      0.06533370899961785,
      0.2514351442503589,
      0.35271572559067943,
      0.23400122550752556,
      0.33747196917113453,
      0.07581107973066051,
      0.17110128476868686,
      0.2600409450936177,
      0.35717794384660767,
      0.2116798262910878,
    ];
    utils.checkValue(y, expected);
  });
});
