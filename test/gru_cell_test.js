'use strict';

import {gruCell} from '../src/gru_cell.js';
import {Tensor} from '../src/tensor.js';
import {sigmoid} from '../src/sigmoid.js';
import {tanh} from '../src/tanh.js';
import * as utils from './utils.js';

describe('test gruCell', function() {
  it('gruCell defaults', function() {
    const batchSize = 3;
    const inputSize = 2;
    const hiddenSize = 5;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(0));
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize);
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      0.12397027,
      0.12397027,
      0.12397027,
      0.12397027,
      0.12397027,
      0.20053662,
      0.20053662,
      0.20053662,
      0.20053662,
      0.20053662,
      0.19991654,
      0.19991654,
      0.19991654,
      0.19991654,
      0.19991654,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with bias', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(0));
    const bias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(0.1));
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize, {bias});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      0.20053662,
      0.20053662,
      0.20053662,
      0.15482337,
      0.15482337,
      0.15482337,
      0.07484276,
      0.07484276,
      0.07484276,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with recurrentBias', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(0));
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {recurrentBias});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      0.14985296,
      0.14985296,
      0.14985296,
      0.0746777,
      0.0746777,
      0.0746777,
      0.03221882,
      0.03221882,
      0.03221882,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with explict resetAfter true', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(2));
    const bias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      1.90645754,
      1.90645754,
      1.90645754,
      1.96068704,
      1.96068704,
      1.96068704,
      1.983688,
      1.983688,
      1.983688,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with resetAfter false', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(2));
    const bias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const resetAfter = false;
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      1.90685618,
      1.90685618,
      1.90685618,
      1.96069813,
      1.96069813,
      1.96069813,
      1.98368835,
      1.98368835,
      1.98368835,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with default zrn layout', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(2));
    const bias = new Tensor([3 * hiddenSize],
        [
          1.9853785,
          2.2497437,
          0.6179927,
          0.3148022,
          -0.4366297,
          -0.9718124,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]);
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with explict zrn layout', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(2));
    const bias = new Tensor([3 * hiddenSize],
        [
          1.9853785,
          2.2497437,
          0.6179927,
          0.3148022,
          -0.4366297,
          -0.9718124,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]);
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const layout = 'zrn';
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter, layout});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with rzn layout', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(2));
    const bias = new Tensor([3 * hiddenSize],
        [
          0.3148022,
          -0.4366297,
          -0.9718124,
          1.9853785,
          2.2497437,
          0.6179927,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]);
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const layout = 'rzn';
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter, layout});
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(output, expected);
  });

  it('gruCell with [tanh, sigmoid] activations', function() {
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([3 * hiddenSize, inputSize],
        new Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([3 * hiddenSize, hiddenSize],
        new Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Array(batchSize * hiddenSize).fill(2));
    const bias = new Tensor([3 * hiddenSize],
        [
          1.9853785,
          2.2497437,
          0.6179927,
          0.3148022,
          -0.4366297,
          -0.9718124,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]);
    const recurrentBias = new Tensor([3 * hiddenSize], new Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const output = gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {
          bias,
          recurrentBias,
          resetAfter,
          activations: [tanh, sigmoid],
        });
    utils.checkShape(output, [batchSize, hiddenSize]);
    const expected = [
      1.99940538,
      1.99962664,
      1.99164689,
      1.99991298,
      1.99994671,
      1.99874425,
      1.99998665,
      1.99999189,
      1.99979985,
    ];
    utils.checkValue(output, expected);
  });
});
