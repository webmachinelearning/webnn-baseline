'use strict';

import {broadcast} from '../src/broadcast.js';
import {Tensor, sizeOfShape} from '../src/tensor.js';
import * as utils from './utils.js';

describe('test broadcast', function() {
  it('broadcast [1] to [3, 4, 5]', function() {
    const inputShape = [1];
    const inputData = [0.6338172];
    const inputTensor = new Tensor(inputShape, inputData);
    const expectedShape = [3, 4, 5];
    const expectedData = new Array(sizeOfShape(expectedShape)).fill(0.6338172);
    const outputTensor = broadcast(inputTensor, [3, 4, 5]);
    utils.checkShape(outputTensor, expectedShape);
    utils.checkValue(outputTensor, expectedData);
  });

  it('broadcast [5] to [3, 4, 5]', function() {
    const inputShape = [5];
    const inputData = [0.6338172, 1.630534, -1.3819867, -1.0427561, 1.058136];
    const inputTensor = new Tensor(inputShape, inputData);
    const expectedShape = [3, 4, 5];
    const expectedData = [
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
      0.6338172,
      1.630534,
      -1.3819867,
      -1.0427561,
      1.058136,
    ];
    const outputTensor = broadcast(inputTensor, [3, 4, 5]);
    utils.checkShape(outputTensor, expectedShape);
    utils.checkValue(outputTensor, expectedData);
  });

  it('broadcast [2, 1, 2] to [2, 2, 2]', function() {
    const inputShape = [2, 1, 2];
    const inputData = [
      0.8189771771430969, 0.9455667734146118, 0.8828932046890259, 0.3519825041294098];
    const inputTensor = new Tensor(inputShape, inputData);
    const expectedShape = [2, 2, 2];
    const expectedData = [
      0.8189771771430969, 0.9455667734146118, 0.8189771771430969, 0.9455667734146118,
      0.8828932046890259, 0.3519825041294098, 0.8828932046890259, 0.3519825041294098,
    ];
    const outputTensor = broadcast(inputTensor, [2, 2, 2]);
    utils.checkShape(outputTensor, expectedShape);
    utils.checkValue(outputTensor, expectedData);
  });
});
