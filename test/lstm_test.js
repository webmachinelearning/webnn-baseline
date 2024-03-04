'use strict';

import {lstm} from '../src/lstm.js';
import {relu} from '../src/relu.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test lstm', function() {
  it('lstm returnSequence=true ' +
  'activations=[relu, relu, relu]', function() {
    const steps = 1;
    const numDirections = 1;
    const batchSize = 2;
    const inputSize = 2;
    const hiddenSize = 2;
    const input = new Tensor([steps, batchSize, inputSize], new Float32Array([1, 2, 2, 1]));
    const weight = new Tensor([numDirections, 4 * hiddenSize, inputSize],
        new Float32Array([
          1, -1, 2, -2, 1, -1, 2, -2,
          1, -1, 2, -2, 1, -1, 2, -2,
        ]));
    const recurrentWeight = new Tensor([numDirections, 4 * hiddenSize, hiddenSize],
        new Array(4 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 4 * hiddenSize],
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const recurrentBias = new Tensor([numDirections, 4 * hiddenSize],
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const peepholeWeight = new Tensor([numDirections, 3 * hiddenSize],
        new Float32Array(3 * hiddenSize).fill(0));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Float32Array(batchSize * hiddenSize).fill(0));
    const initialCellState = new Tensor([numDirections, batchSize, hiddenSize],
        new Float32Array(batchSize * hiddenSize).fill(0));
    const returnSequence = true;
    const activations = [
      relu,
      relu,
      relu,
    ];
    const outputs = lstm(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, peepholeWeight, initialHiddenState,
          initialCellState, returnSequence, activations});
    console.log('outputs: ', outputs);
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    utils.checkShape(outputs[1], [numDirections, batchSize, hiddenSize]);
    utils.checkShape(outputs[2], [steps, numDirections, batchSize, hiddenSize]);
    const expected = [
      [
        1, 8, 27, 216,
      ],
      [
        1, 4, 9, 36,
      ],
      [
        1, 8, 27, 216,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(outputs[i], expected[i]);
    }
  });

  it('lstm steps=2 direction="backward" returnSequence=true' +
  'activations=[relu, relu, relu]', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 2;
    const inputSize = 2;
    const hiddenSize = 2;
    const input = new Tensor([steps, batchSize, inputSize],
        new Float32Array([1, 2, 2, 1, 3, 4, 1, 2]));
    const weight = new Tensor([numDirections, 4 * hiddenSize, inputSize],
        new Float32Array([
          1, -1, 2, -2, 1, -1, 2, -2,
          1, -1, 2, -2, 1, -1, 2, -2,
        ]));
    const recurrentWeight = new Tensor([numDirections, 4 * hiddenSize, hiddenSize],
        new Array(4 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 4 * hiddenSize],
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const recurrentBias = new Tensor([numDirections, 4 * hiddenSize],
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const peepholeWeight = new Tensor([numDirections, 3 * hiddenSize],
        new Float32Array(3 * hiddenSize).fill(0));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Float32Array(batchSize * hiddenSize).fill(0));
    const initialCellState = new Tensor([numDirections, batchSize, hiddenSize],
        new Float32Array(batchSize * hiddenSize).fill(0));
    const returnSequence = true;
    const direction = 'backward';
    const activations = [
      relu,
      relu,
      relu,
    ];
    const outputs = lstm(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, peepholeWeight, initialHiddenState,
          initialCellState, direction, returnSequence, activations});
    console.log('outputs: ', outputs);
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    utils.checkShape(outputs[1], [numDirections, batchSize, hiddenSize]);
    utils.checkShape(outputs[2], [steps, numDirections, batchSize, hiddenSize]);
    const expected = [
      [10.469, 58.02899999999999, 74.529, 518.9490000000001],
      [5.51, 20.009999999999998, 19.11, 75.21000000000001],
      [
        1,
        8,
        1,
        8,
        10.469,
        58.02899999999999,
        74.529,
        518.9490000000001,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(outputs[i], expected[i]);
    }
  });
});
