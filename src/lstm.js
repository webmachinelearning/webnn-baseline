'use strict';

import {concat} from './concat.js';
import {lstmCell} from './lstm_cell.js';
import {reshape} from './reshape.js';
import {sizeOfShape, Tensor} from './lib/tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
import {squeeze} from './squeeze.js';
import {tanh} from './tanh.js';
import {validateLstmParams} from './lib/validate-input.js';

/**
 *Long Short-Term Memory [LSTM] recurrent network uses an input, output, forget,
 *and cell gate to compute the output state that rolls into the output across the
 * temporal sequence of the network.
 * @param {Tensor} input
 * @param {Tensor} weight
 * @param {Tensor} recurrentWeight
 * @param {Number} steps
 * @param {Number} hiddenSize
 * @param {MLLstmOptions} options
 * @return {Array.<Tensor>}
 */

export function lstm(input, weight, recurrentWeight, steps, hiddenSize,
    {bias, recurrentBias, peepholeWeight, initialHiddenState,
      initialCellState, returnSequence = false, direction = 'forward', layout = 'iofg',
      activations = [sigmoid, tanh, tanh]}={}) {
  validateLstmParams(...arguments);
  const numDirections = (direction == 'both' ? 2 : 1);
  const batchSize = input.shape[1];
  const inputSize = input.shape[2];

  let hiddenState;
  let cellState;
  if (initialHiddenState) {
    hiddenState = initialHiddenState;
  } else {
    const initialHiddenStateShape = [numDirections, batchSize, hiddenSize];
    hiddenState = new Tensor(
        initialHiddenStateShape, new Array(sizeOfShape(initialHiddenStateShape)).fill(0));
  }
  if (initialCellState) {
    cellState = initialCellState;
  } else {
    const initialCellState = [numDirections, batchSize, hiddenSize];
    cellState = new Tensor(
        initialCellState, new Array(sizeOfShape(initialCellState)).fill(0));
  }

  let sequence;
  const currentWeight = [];
  const currentRecurrentWeight = [];
  const currentBias = [];
  const currentRecurrentBias = [];
  const currentPeepholeWeight = [];

  for (let dir = 0; dir < numDirections; ++dir) {
    currentWeight.push(squeeze(slice(weight, [dir, 0, 0], [1, 4 * hiddenSize, inputSize])));
    currentRecurrentWeight.push(squeeze(slice(recurrentWeight,
        [dir, 0, 0], [1, 4 * hiddenSize, hiddenSize])));
    currentBias.push(bias ? (squeeze(slice(bias, [dir, 0], [1, 4 * hiddenSize]))) : null);
    currentRecurrentBias.push(recurrentBias ?
      (squeeze(slice(recurrentBias, [dir, 0], [1, 4 * hiddenSize]))) : null);
    currentPeepholeWeight.push(peepholeWeight ?
      (squeeze(slice(peepholeWeight, [dir, 0], [1, 3 * hiddenSize]))) : null);
  }

  for (let step = 0; step < steps; ++step) {
    const currentHidden = [];
    const currentCell = [];
    let nextHidden = null;
    let nextCell = null;

    for (let dir = 0; dir < numDirections; ++dir) {
      currentHidden.push(squeeze(slice(hiddenState, [dir, 0, 0], [1, batchSize, hiddenSize])));
      currentCell.push(squeeze(slice(cellState, [dir, 0, 0], [1, batchSize, hiddenSize])));
    }

    for (let dir = 0; dir < numDirections; ++dir) {
      const slice0 = (dir == 1 || direction == 'backward' ? steps - step - 1 : step);
      const currentInput = squeeze(slice(input, [slice0, 0, 0], [1, batchSize, inputSize]));

      const results = lstmCell(
          currentInput, currentWeight[dir], currentRecurrentWeight[dir],
          currentHidden[dir], currentCell[dir], hiddenSize, {bias: currentBias[dir],
            recurrentBias: currentRecurrentBias[dir], peepholeWeight: currentPeepholeWeight[dir],
            layout: layout, activations: activations});

      const output = reshape(results[0], [1, null, hiddenSize]);
      const cell = reshape(results[1], [1, null, hiddenSize]);

      nextHidden = (nextHidden ? concat([nextHidden, output], 0) : output);
      nextCell = (nextCell ? concat([nextCell, cell], 0) : cell);
    }

    hiddenState = nextHidden;
    cellState = nextCell;

    if (returnSequence) {
      nextHidden = reshape(nextHidden, [1, numDirections, null, hiddenSize]);
      sequence = (sequence ? concat([sequence, nextHidden], 0) : nextHidden);
    }
  }

  return (sequence ? [hiddenState, cellState, sequence] : [hiddenState, cellState]);
}
