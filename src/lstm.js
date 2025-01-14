'use strict';

import {concat} from './concat.js';
import {lstmCell} from './lstm_cell.js';
import {reshape, squeeze} from './reshape.js';
import {sizeOfShape, Tensor} from './lib/tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
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
      activations = [sigmoid, tanh, tanh]} = {}) {
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

  const currentWeight = [];
  const currentRecurrentWeight = [];
  const currentBias = [];
  const currentRecurrentBias = [];
  const currentPeepholeWeight = [];
  let forwardSequence = null;
  let backwardSequence = null;
  let outputHidden = null;
  let outputCell = null;

  for (let dir = 0; dir < numDirections; ++dir) {
    currentWeight.push(squeeze(slice(weight, [dir, 0, 0], [1, 4 * hiddenSize, inputSize])));
    currentRecurrentWeight.push(squeeze(slice(recurrentWeight,
        [dir, 0, 0], [1, 4 * hiddenSize, hiddenSize])));
    currentBias.push(bias ? (squeeze(slice(bias, [dir, 0], [1, 4 * hiddenSize]))) : null);
    currentRecurrentBias.push(recurrentBias ?
      (squeeze(slice(recurrentBias, [dir, 0], [1, 4 * hiddenSize]))) : null);
    currentPeepholeWeight.push(peepholeWeight ?
      (squeeze(slice(peepholeWeight, [dir, 0], [1, 3 * hiddenSize]))) : null);

    let currentHidden = squeeze(slice(hiddenState, [dir, 0, 0], [1, batchSize, hiddenSize]));
    let currentCell = squeeze(slice(cellState, [dir, 0, 0], [1, batchSize, hiddenSize]));

    for (let step = 0; step < steps; ++step) {
      const slice0 = dir === 1 || direction === 'backward' ? steps - step - 1 : step;
      const currentInput = squeeze(slice(input, [slice0, 0, 0], [1, batchSize, inputSize]));

      [currentHidden, currentCell] = lstmCell(
          currentInput, currentWeight[dir], currentRecurrentWeight[dir],
          currentHidden, currentCell, hiddenSize, {bias: currentBias[dir],
            recurrentBias: currentRecurrentBias[dir], peepholeWeight: currentPeepholeWeight[dir],
            layout: layout, activations: activations});

      if (returnSequence) {
        // Expand hidden of 2D([batchSize, hiddenSize]) to
        // 4D([steps, numDirections, batchSize, hiddenSize])
        const expandedHiddenAs4D = reshape(currentHidden, [1, 1, batchSize, hiddenSize]);
        if (direction === 'forward' || (dir === 0 && direction === 'both')) {
          forwardSequence = forwardSequence ?
              concat([forwardSequence, expandedHiddenAs4D], 0) :
              expandedHiddenAs4D;
        } else if (direction === 'backward' || (dir === 1 && direction === 'both')) {
          backwardSequence = backwardSequence ?
              concat([expandedHiddenAs4D, backwardSequence], 0) :
              expandedHiddenAs4D;
        }
      }
    }

    // Expand hidden of 2D([batchSize, hiddenSize]) to 3D([numDirections, batchSize, hiddenSize])
    const expandHiddenAs3D = reshape(currentHidden, [1, batchSize, hiddenSize]);
    // Concat along axis 0 (numDirections dimension)
    outputHidden = outputHidden ? concat([outputHidden, expandHiddenAs3D], 0) : expandHiddenAs3D;

    // Expand cell of 2D([batchSize, hiddenSize]) to 3D([numDirections, batchSize, hiddenSize])
    const expandCellAs3D = reshape(currentCell, [1, batchSize, hiddenSize]);
    // Concat along axis 0 (numDirections dimension)
    outputCell = outputCell ? concat([outputCell, expandCellAs3D], 0) : expandCellAs3D;
  }

  if (returnSequence) {
    // outputSequence: [steps, numDirections, batchSize, hiddenSize]
    let outputSequence;
    if (direction === 'forward') {
      outputSequence = forwardSequence;
    } else if (direction === 'backward') {
      outputSequence = backwardSequence;
    } else if (direction === 'both') {
      // Concat along axis 1 (numDirections dimension)
      outputSequence = concat([forwardSequence, backwardSequence], 1);
    }
    return [outputHidden, outputCell, outputSequence];
  } else {
    return [outputHidden, outputCell];
  }
}
