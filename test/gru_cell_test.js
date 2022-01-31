'use strict';

import {gruCell} from '../src/gru_cell.js';
import {Tensor} from '../src/lib/tensor.js';
import {sigmoid} from '../src/lib/sigmoid.js';
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
      0.12397026217591961,
      0.12397026217591961,
      0.12397026217591961,
      0.12397026217591961,
      0.12397026217591961,
      0.20053661855501925,
      0.20053661855501925,
      0.20053661855501925,
      0.20053661855501925,
      0.20053661855501925,
      0.19991654116571125,
      0.19991654116571125,
      0.19991654116571125,
      0.19991654116571125,
      0.19991654116571125,
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
      0.20053661855501925,
      0.20053661855501925,
      0.20053661855501925,
      0.15482337214048048,
      0.15482337214048048,
      0.15482337214048048,
      0.07484276504070396,
      0.07484276504070396,
      0.07484276504070396,
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
      0.14985295238282167,
      0.14985295238282167,
      0.14985295238282167,
      0.07467770390292117,
      0.07467770390292117,
      0.07467770390292117,
      0.032218815985522856,
      0.032218815985522856,
      0.032218815985522856,
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
      1.9064574801795497,
      1.9064574801795497,
      1.9064574801795497,
      1.9606870240735346,
      1.9606870240735346,
      1.9606870240735346,
      1.9836880687096186,
      1.9836880687096186,
      1.9836880687096186,
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
      1.906856117423314,
      1.906856117423314,
      1.906856117423314,
      1.9606980991458889,
      1.9606980991458889,
      1.9606980991458889,
      1.983688371193181,
      1.983688371193181,
      1.983688371193181,
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
      1.9801673183552388,
      1.9812534682811542,
      1.9376592706336329,
      1.9935192730591977,
      1.9947569570033654,
      1.9759958501762682,
      1.997469445392646,
      1.9980404252433588,
      1.9902071255213296,
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
      1.9801673183552388,
      1.9812534682811542,
      1.9376592706336329,
      1.9935192730591977,
      1.9947569570033654,
      1.9759958501762682,
      1.997469445392646,
      1.9980404252433588,
      1.9902071255213296,
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
      1.9801673183552388,
      1.9812534682811542,
      1.9376592706336329,
      1.9935192730591977,
      1.9947569570033654,
      1.9759958501762682,
      1.997469445392646,
      1.9980404252433588,
      1.9902071255213296,
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
      1.9994052973405467,
      1.9996265670444457,
      1.9916469375315222,
      1.9999129608020485,
      1.9999467564798181,
      1.9987442445027492,
      1.9999865812225888,
      1.99999193815786,
      1.9997998773572325,
    ];
    utils.checkValue(output, expected);
  });
});
