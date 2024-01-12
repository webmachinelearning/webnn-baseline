'use strict';
import {l2Pool2d} from '../src/pool2d.js';
import {Tensor} from '../src/lib/tensor.js';

import * as utils from './utils.js';

describe('test pool2d', function() {
  it('l2Pool2d default', function() {
    const x = new Tensor([1, 1, 1, 1], [2]);
    const windowDimensions = [1, 1];
    const y = l2Pool2d(x, {windowDimensions});
    utils.checkShape(y, [1, 1, 1, 1]);
    utils.checkValue(y, [2]);
  });

  it('l2Pool2d default', function() {
    const x = new Tensor([1, 1, 4, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const windowDimensions = [3, 3];
    const y = l2Pool2d(x, {windowDimensions});
    utils.checkShape(y, [1, 1, 2, 2]);
    utils.checkValue(y, [
      20.639767440550294,
      23.302360395462088,
      31.654383582688826,
      34.51086785347479,
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
      23.302360395462088,
      31.654383582688826,
      34.51086785347479,
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
      16.24807680927192,
      21.633307652783937,
      23.49468024894146,
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
      16.24807680927192,
      21.633307652783937,
      23.49468024894146,
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
      35.21363372331802,  32.89376840679705,
      29.748949561287034,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.558046773455466, 44.384682042344295,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.33739825307956,
      60.8276253029822, 53.907327887774215,
      64.18722614352485,  73.95944834840239,
      67.94115100585212,  60.41522986797286,
      52.507142371300304, 62.369864518050704,
      71.69379331573968,   65.7419196555744,
      58.35237784358063,
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
      35.21363372331802,  32.89376840679705,
      29.748949561287034,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.558046773455466, 44.384682042344295,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.33739825307956,
      60.8276253029822, 53.907327887774215,
      64.18722614352485,  73.95944834840239,
      67.94115100585212,  60.41522986797286,
      52.507142371300304, 62.369864518050704,
      71.69379331573968,   65.7419196555744,
      58.35237784358063,
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
      35.21363372331802,  32.89376840679705,
      29.748949561287034,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.558046773455466, 44.384682042344295,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.33739825307956,
      60.8276253029822, 53.907327887774215,
      64.18722614352485,  73.95944834840239,
      67.94115100585212,  60.41522986797286,
      52.507142371300304, 62.369864518050704,
      71.69379331573968,   65.7419196555744,
      58.35237784358063,
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
      24.899799195977465, 24.879710609249457,
      40.54626986542659, 60.860496218811754,
      67.82329983125268, 63.324560795950255,
      76.81145747868608,  112.5344391730816,
      120.2331069215131,  109.1146186356347,
      90.50414355155237,  131.4610208388783,
      138.31124321616085, 124.21352583354198,
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
      24.899799195977465, 24.879710609249457,
      40.54626986542659, 60.860496218811754,
      67.82329983125268, 63.324560795950255,
      76.81145747868608,  112.5344391730816,
      120.2331069215131,  109.1146186356347,
      90.50414355155237,  131.4610208388783,
      138.31124321616085, 124.21352583354198,
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
      35.21363372331802,  32.89376840679705,
      29.748949561287034,  38.28837943815329,
      46.04345773288535,   53.5723809439155,
      49.558046773455466, 44.384682042344295,
      54.037024344425184,  64.42049363362563,
      74.33034373659252,  68.33739825307956,
      60.8276253029822, 53.907327887774215,
      64.18722614352485,  73.95944834840239,
      67.94115100585212,  60.41522986797286,
      52.507142371300304, 62.369864518050704,
      71.69379331573968,   65.7419196555744,
      58.35237784358063,
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
      13.038404810405298,
      28.460498941515414,
      32.4037034920393,
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
      13.038404810405298,
      28.460498941515414,
      32.4037034920393,
    ];
    utils.checkValue(y, expected);
  });
});
