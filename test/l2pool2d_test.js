'use strict';
import {l2Pool2d} from '../src/pool2d.js';
import {Tensor} from '../src/lib/tensor.js';

import * as utils from './utils.js';

describe('test pool2d', function() {
  it('l2Pool2d default', function() {
    const x = new Tensor([1, 1, 4, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const y = l2Pool2d(x, {windowDimensions});
    utils.checkShape(y, [1, 1, 2, 2]);
    utils.checkValue(y, [
      20.639767440550294,
      23.259406699226016,
      31.336879231984796,
      34.07345007480164,
    ]);
  });

  it('l2Pool2d nhwc', function() {
    const x = new Tensor([1, 4, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const layout = 'nhwc';
    const y = l2Pool2d(x, {windowDimensions, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    utils.checkValue(y, [
      20.639767440550294,
      23.259406699226016,
      31.336879231984796,
      34.07345007480164,
    ]);
  });

  it('l2Pool2d dilations default', function() {
    const x = new Tensor([1, 1, 4, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [2, 2];
    const dilations = [2, 2];
    const y = l2Pool2d(x, {windowDimensions, dilations});
    utils.checkShape(y, [1, 1, 2, 2]);
    utils.checkValue(y, [
      14.560219778561036,
      16.186414056238647,
      21.166010488516726,
      22.847319317591726,
    ]);
  });

  it('l2Pool2d dilations nhwc', function() {
    const x = new Tensor([1, 4, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [2, 2];
    const dilations = [2, 2];
    const layout = 'nhwc';
    const y = l2Pool2d(x, {windowDimensions, dilations, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    utils.checkValue(y, [
      14.560219778561036,
      16.186414056238647,
      21.166010488516726,
      22.847319317591726,
    ]);
  });

  it('l2Pool2d pads default', function() {
    const x = new Tensor([1, 1, 5, 5], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const y = l2Pool2d(x, {windowDimensions, padding});
    utils.checkShape(y, [1, 1, 5, 5]);
    const expected = [
      24.43358344574123, 29.832867780352597,
      35.21363372331802, 32.863353450309965,
      29.647934160747187,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.53786430600334,  44.31703961232068,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.32276341015489,
      60.778285596090974,  53.62835071116769,
      63.953107821277925,   73.7563556583431,
      67.63135367564367,  59.94997914928745,
      51.44900387762624,  61.48170459575759,
      70.92249290598858,  64.73020933072904,
      57,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d pads nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const layout = 'nhwc';
    const y = l2Pool2d(x, {windowDimensions, padding, layout});
    utils.checkShape(y, [1, 5, 5, 1]);
    const expected = [
      24.43358344574123, 29.832867780352597,
      35.21363372331802, 32.863353450309965,
      29.647934160747187,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.53786430600334,  44.31703961232068,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.32276341015489,
      60.778285596090974,  53.62835071116769,
      63.953107821277925,   73.7563556583431,
      67.63135367564367,  59.94997914928745,
      51.44900387762624,  61.48170459575759,
      70.92249290598858,  64.73020933072904,
      57,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d autoPad same-upper default', function() {
    const x = new Tensor([1, 1, 5, 5], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const autoPad = 'same-upper';
    const y = l2Pool2d(x, {windowDimensions, autoPad});
    utils.checkShape(y, [1, 1, 5, 5]);
    const expected = [
      24.43358344574123, 29.832867780352597,
      35.21363372331802, 32.863353450309965,
      29.647934160747187,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.53786430600334,  44.31703961232068,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.32276341015489,
      60.778285596090974,  53.62835071116769,
      63.953107821277925,   73.7563556583431,
      67.63135367564367,  59.94997914928745,
      51.44900387762624,  61.48170459575759,
      70.92249290598858,  64.73020933072904,
      57,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d autoPad explicit nhwc', function() {
    const x = new Tensor([1, 7, 7, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const padding = [2, 1, 2, 1];
    const strides = [2, 2];
    const autoPad = 'explicit';
    const layout = 'nhwc';
    const y = l2Pool2d(
        x, {windowDimensions, autoPad, padding, strides, layout});
    utils.checkShape(y, [1, 4, 4, 1]);
    const expected = [
      12.24744871391589,   19.8997487421324,
      24.779023386727733, 24.474476501040833,
      40.54626986542659, 60.860496218811754,
      67.77905281132217, 63.166446789415026,
      75.43208866258443, 111.59749101122301,
      119.09659944767525, 107.53604047016051,
      85.901105930017, 128.33549781724463,
      134.90737563232042,  119.8874472161285,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d autoPad same-lower nhwc', function() {
    const x = new Tensor([1, 7, 7, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ]);
    const windowDimensions = [4, 4];
    const strides = [2, 2];
    const autoPad = 'same-lower';
    const layout = 'nhwc';
    const y =
        l2Pool2d(x, {windowDimensions, autoPad, strides, layout});
    utils.checkShape(y, [1, 4, 4, 1]);
    const expected = [
      12.24744871391589,   19.8997487421324,
      24.779023386727733, 24.474476501040833,
      40.54626986542659, 60.860496218811754,
      67.77905281132217, 63.166446789415026,
      75.43208866258443, 111.59749101122301,
      119.09659944767525, 107.53604047016051,
      85.901105930017, 128.33549781724463,
      134.90737563232042,  119.8874472161285,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d autoPad same-upper nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [5, 5];
    const autoPad = 'same-upper';
    const layout = 'nhwc';
    const y = l2Pool2d(x, {windowDimensions, autoPad, layout});

    utils.checkShape(y, [1, 5, 5, 1]);
    const expected = [
      24.43358344574123, 29.832867780352597,
      35.21363372331802, 32.863353450309965,
      29.647934160747187,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.53786430600334,  44.31703961232068,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.32276341015489,
      60.778285596090974,  53.62835071116769,
      63.953107821277925,   73.7563556583431,
      67.63135367564367,  59.94997914928745,
      51.44900387762624,  61.48170459575759,
      70.92249290598858,  64.73020933072904,
      57,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d strides default', function() {
    const x = new Tensor([1, 1, 5, 5], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const y = l2Pool2d(x, {windowDimensions, strides});
    utils.checkShape(y, [1, 1, 2, 2]);
    const expected = [
      9.486832980505138,
      12.806248474865697,
      26.457513110645905,
      29.899832775452108,
    ];
    utils.checkValue(y, expected);
  });

  it('l2Pool2d strides nhwc', function() {
    const x = new Tensor([1, 5, 5, 1], [
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]);
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const layout = 'nhwc';
    const y = l2Pool2d(x, {windowDimensions, strides, layout});
    utils.checkShape(y, [1, 2, 2, 1]);
    const expected = [
      9.486832980505138,
      12.806248474865697,
      26.457513110645905,
      29.899832775452108,
    ];
    utils.checkValue(y, expected);
  });
});
