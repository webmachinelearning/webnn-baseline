'use strict';

import {slice} from '../src/slice.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test slice', function() {
  function testSlice(inputShape, inputData, starts, sizes, expectedShape, expected, options = {}) {
    const input = new Tensor(inputShape, inputData);
    const output = slice(input, starts, sizes, options);
    utils.checkShape(output, expectedShape);
    utils.checkValue(output, expected);
  }

  it('slice 1D default options', function() {
    const inputShape = [4];
    const inputData = [0, 1, 2, 3];
    const starts = [3];
    const sizes = [1];
    const expectedShape = [1];
    const expected = [3];
    testSlice(inputShape, inputData, starts, sizes, expectedShape, expected);
  });

  it('slice 3D default options', function() {
    const inputShape = [3, 4, 5];
    const inputData = [
      1.3165863e+00,  4.1239005e-02,  4.6697399e-01,  -6.6145003e-02,
      -3.7128052e-01, -1.0660021e+00, 7.5784922e-01,  3.5759725e-02,
      1.9211160e+00,  -8.1603736e-01, 1.1800343e-01,  -1.8293047e+00,
      -2.1316205e-01, -3.6369815e-01, 6.4205879e-01,  7.1544610e-02,
      6.8498695e-01,  1.0001093e+00,  -5.6261641e-01, -7.3343945e-01,
      1.6827687e+00,  1.2653192e+00,  5.8872145e-01,  3.1535852e-01,
      3.5038650e-01,  3.5865438e-01,  -3.6469769e-01, -8.7751287e-01,
      2.7995768e-01,  -1.6042528e+00, 8.6336482e-01,  -1.7991974e+00,
      -6.8652731e-01, 1.3729302e-03,  -7.7775210e-01, 1.0199220e-01,
      4.2299256e-01,  1.1432177e-01,  -5.0116669e-02, 1.5525131e+00,
      -8.7060851e-01, 4.5739245e-01,  1.3543987e-01,  -1.5927458e-02,
      9.1792661e-01,  -4.5001405e-01, 1.9954188e-01,  -5.1338053e-01,
      -4.1026011e-01, -1.2718531e+00, 4.2538303e-01,  -1.5449624e-01,
      -3.4380481e-01, 7.8374326e-01,  1.7837452e+00,  9.6105379e-01,
      -4.8783422e-01, -9.4987392e-01, -8.8750905e-01, -9.8019439e-01,
    ];
    const starts = [0, 0, 1];
    const sizes = [2, 3, 4];
    const expectedShape = [2, 3, 4];
    const expected = [
      4.1239005e-02,  4.6697399e-01,  -6.6145003e-02, -3.7128052e-01,
      7.5784922e-01,  3.5759725e-02,  1.9211160e+00,  -8.1603736e-01,
      -1.8293047e+00, -2.1316205e-01, -3.6369815e-01, 6.4205879e-01,
      1.2653192e+00,  5.8872145e-01,  3.1535852e-01,  3.5038650e-01,
      -3.6469769e-01, -8.7751287e-01, 2.7995768e-01,  -1.6042528e+00,
      -1.7991974e+00, -6.8652731e-01, 1.3729302e-03,  -7.7775210e-01,
    ];
    testSlice(inputShape, inputData, starts, sizes, expectedShape, expected);
  });

  it('slice 2D with strides=[1, 5]', function() {
    const inputShape = [3, 20];
    const inputData = [
      1.3165863e+00,  4.1239005e-02,  4.6697399e-01,  -6.6145003e-02,
      -3.7128052e-01, -1.0660021e+00, 7.5784922e-01,  3.5759725e-02,
      1.9211160e+00,  -8.1603736e-01, 1.1800343e-01,  -1.8293047e+00,
      -2.1316205e-01, -3.6369815e-01, 6.4205879e-01,  7.1544610e-02,
      6.8498695e-01,  1.0001093e+00,  -5.6261641e-01, -7.3343945e-01,

      1.6827687e+00,  1.2653192e+00,  5.8872145e-01,  3.1535852e-01,
      3.5038650e-01,  3.5865438e-01,  -3.6469769e-01, -8.7751287e-01,
      2.7995768e-01,  -1.6042528e+00, 8.6336482e-01,  -1.7991974e+00,
      -6.8652731e-01, 1.3729302e-03,  -7.7775210e-01, 1.0199220e-01,
      4.2299256e-01,  1.1432177e-01,  -5.0116669e-02, 1.5525131e+00,

      -8.7060851e-01, 4.5739245e-01,  1.3543987e-01,  -1.5927458e-02,
      9.1792661e-01,  -4.5001405e-01, 1.9954188e-01,  -5.1338053e-01,
      -4.1026011e-01, -1.2718531e+00, 4.2538303e-01,  -1.5449624e-01,
      -3.4380481e-01, 7.8374326e-01,  1.7837452e+00,  9.6105379e-01,
      -4.8783422e-01, -9.4987392e-01, -8.8750905e-01, -9.8019439e-01,
    ];
    const starts = [1, 4];
    const sizes = [2, 10];
    const strides = [1, 3];
    const expectedShape = [2, 4];
    const expected = [
      3.5038650e-01, -8.7751287e-01, 8.6336482e-01, 1.3729302e-03,
      9.1792661e-01, -5.1338053e-01, 4.2538303e-01, 7.8374326e-01,
    ];
    testSlice(inputShape, inputData, starts, sizes, expectedShape, expected, {strides});
  });
});
