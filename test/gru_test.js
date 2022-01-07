'use strict';

import {gru} from '../src/gru.js';
import {Tensor} from '../src/tensor.js';
import * as utils from './utils.js';

describe('test gru', function() {
  it('gru with 1 step', function() {
    const steps = 1;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([steps, batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
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
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(2));
    const resetAfter = true;
    const layout = 'rzn';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, resetAfter, layout});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
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
    utils.checkValue(outputs[0], expected);
  });

  it('gru with 2 steps', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with explict returnSequence false', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const returnSequence = false;
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, returnSequence});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with returnSequence true', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const returnSequence = true;
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, returnSequence});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    utils.checkShape(outputs[1], [steps, numDirections, batchSize, hiddenSize]);
    const expected = [
      [
        0.22391089,
        0.22391089,
        0.22391089,
        0.22391089,
        0.22391089,
        0.1653014,
        0.1653014,
        0.1653014,
        0.1653014,
        0.1653014,
        0.0797327,
        0.0797327,
        0.0797327,
        0.0797327,
        0.0797327,
      ],
      [
        0.20053661,
        0.20053661,
        0.20053661,
        0.20053661,
        0.20053661,
        0.15482338,
        0.15482338,
        0.15482338,
        0.15482338,
        0.15482338,
        0.07484276,
        0.07484276,
        0.07484276,
        0.07484276,
        0.07484276,
        0.22391089,
        0.22391089,
        0.22391089,
        0.22391089,
        0.22391089,
        0.1653014,
        0.1653014,
        0.1653014,
        0.1653014,
        0.1653014,
        0.0797327,
        0.0797327,
        0.0797327,
        0.0797327,
        0.0797327,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(outputs[i], expected[i]);
    }
  });

  it('gru with explict forward direction', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'forward';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with backward direction', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'backward';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with both direction', function() {
    const steps = 2;
    const numDirections = 2;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'both';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.2239109,
      0.2239109,
      0.2239109,
      0.2239109,
      0.2239109,
      0.16530138,
      0.16530138,
      0.16530138,
      0.16530138,
      0.16530138,
      0.07973271,
      0.07973271,
      0.07973271,
      0.07973271,
      0.07973271,
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru without initialHiddenState', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
    ];
    utils.checkValue(outputs[0], expected);
  });
});
