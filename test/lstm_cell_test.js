'use strict';

import {lstmCell} from '../src/lstm_cell.js';
import {relu} from '../src/relu.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test lstmCell', function() {
  it('lstmCell lstmCell activations=[relu, relu, relu]', function() {
    const batchSize = 2;
    const inputSize = 2;
    const hiddenSize = 2;
    const input = new Tensor([batchSize, inputSize], [1, 2, 2, 1]);
    const weight = new Tensor([4 * hiddenSize, inputSize],
        new Float32Array([
          1, -1, 2, -2, 1, -1, 2, -2,
          1, -1, 2, -2, 1, -1, 2, -2,
        ]));
    const recurrentWeight = new Tensor([4 * hiddenSize, hiddenSize],
        new Float32Array(4 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = new Tensor([batchSize, hiddenSize],
        new Float32Array(batchSize * hiddenSize).fill(0));
    const cellState = new Tensor([batchSize, hiddenSize],
        new Float32Array(batchSize * hiddenSize).fill(0));
    const bias = new Tensor([4* hiddenSize],
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const recurrentBias = new Tensor([4* hiddenSize],
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const peepholeWeight = new Tensor([3* hiddenSize],
        new Float32Array(3 * hiddenSize).fill(0));
    const activations = [
      relu,
      relu,
      relu,
    ];
    const outputs = lstmCell(
        input, weight, recurrentWeight, hiddenState, cellState, hiddenSize,
        {bias, recurrentBias, peepholeWeight, activations});
    utils.checkShape(outputs[0], [batchSize, hiddenSize]);
    utils.checkShape(outputs[1], [batchSize, hiddenSize]);
    const expected = [
      [
        1, 8, 27, 216,
      ],
      [
        1, 4, 9, 36,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(outputs[i], expected[i]);
    }
  });
});
