'use strict';

import {softmax} from '../src/softmax.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test softmax', function() {
  it('softmax 1D axis=0', function() {
    const x = new Tensor([12], [
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
    const y = softmax(x, 0);
    utils.checkShape(y, [12]);
    const expected = [
      0.11739878814610716,
      0.13197030364705775,
      0.02384582107272479,
      0.09177004570831693,
      0.13690530122809944,
      0.09082670814351343,
      0.13098849368862656,
      0.029425789535106012,
      0.042239560094127944,
      0.06419598275994867,
      0.0881760721072441,
      0.05225713386912745,
    ];
    utils.checkValue(y, expected);
  });

  it('softmax 2D axis=1', function() {
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
    const y = softmax(x, 1);
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

  it('softmax 2D axis=0', function() {
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
    const y = softmax(x, 0);
    utils.checkShape(y, [3, 4]);
    const expected = [
      0.39589041396296437,
      0.4598380662696866,
      0.09812675655555042,
      0.5290773987775849,
      0.4616699817161939,
      0.3164770913163354,
      0.5390242589073232,
      0.16964707889210112,
      0.14243960432084168,
      0.22368484241397796,
      0.3628489845371264,
      0.30127552233031407,
    ];
    utils.checkValue(y, expected);
  });

  it('softmax 3D axis=1', function() {
    const x = new Tensor([3, 2, 2], [
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
    const y = softmax(x, 1);
    utils.checkShape(y, [3, 2, 2]);
    const expected = [
      0.8311735845735523,
      0.5898368534208582,
      0.16882641542644772,
      0.4101631465791417,
      0.511043196318362,
      0.7552999721157738,
      0.48895680368163796,
      0.24470002788422615,
      0.3238841799954374,
      0.5512603236238347,
      0.6761158200045627,
      0.44873967637616524,
    ];
    utils.checkValue(y, expected);
  });

  it('softmax 3D axis=2', function() {
    const x = new Tensor([3, 2, 2], [
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
    const y = softmax(x, 2);
    utils.checkShape(y, [3, 2, 2]);
    const expected = [
      0.4707832366149116,
      0.5292167633850885,
      0.20625041991757956,
      0.7937495800824204,
      0.6011684593916593,
      0.39883154060834064,
      0.8165637813307101,
      0.1834362186692899,
      0.39685577732279226,
      0.6031442226772078,
      0.6278862003768544,
      0.37211379962314556,
    ];
    utils.checkValue(y, expected);
  });
});
