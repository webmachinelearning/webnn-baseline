'use strict';

import {concat} from './concat.js';
import {gruCell} from './gru_cell.js';
import {reshape, squeeze} from './reshape.js';
import {sizeOfShape, Tensor} from './lib/tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
import {tanh} from './tanh.js';
import {validateGruParams} from './lib/validate-input.js';

/**
 * Gated Recurrent Unit [GRU] recurrent network using an update gate and a reset gate to compute
 * the hidden state that rolls into the output across the temporal sequence of the Network
 * @param {Tensor} input
 * @param {Tensor} weight
 * @param {Tensor} recurrentWeight
 * @param {Number} steps
 * @param {Number} hiddenSize
 * @param {MLGruOptions} options
 * @return {Array.<Tensor>}
 */
export function gru(input, weight, recurrentWeight, steps, hiddenSize,
    {bias, recurrentBias, initialHiddenState, resetAfter = true,
      returnSequence = false, direction = 'forward',
      layout = 'zrn', activations = [sigmoid, tanh]} = {}) {
  validateGruParams(...arguments);
  const numDirections = (direction === 'both' ? 2 : 1);
  const batchSize = input.shape[1];
  const inputSize = input.shape[2];

  let hiddenState;
  if (initialHiddenState) {
    hiddenState = initialHiddenState;
  } else {
    const initialHiddenStateShape = [numDirections, batchSize, hiddenSize];
    hiddenState = new Tensor(
        initialHiddenStateShape, new Array(sizeOfShape(initialHiddenStateShape)).fill(0));
  }

  const currentWeight = [];
  const currentRecurrentWeight = [];
  const currentBias = [];
  const currentRecurrentBias = [];
  let forwardSequence = null;
  let backwardSequence = null;
  let outputHidden = null;

  for (let dir = 0; dir < numDirections; ++dir) {
    currentWeight.push(squeeze(slice(weight, [dir, 0, 0], [1, 3 * hiddenSize, inputSize])));
    currentRecurrentWeight.push(squeeze(
        slice(recurrentWeight, [dir, 0, 0], [1, 3 * hiddenSize, hiddenSize])));
    currentBias.push(
        bias ? (squeeze(slice(bias, [dir, 0], [1, 3 * hiddenSize]))) : null);
    currentRecurrentBias.push(
        recurrentBias ? (squeeze(slice(recurrentBias, [dir, 0], [1, 3 * hiddenSize]))) : null);
    let currentHidden = squeeze(slice(hiddenState, [dir, 0, 0], [1, batchSize, hiddenSize]));

    for (let step = 0; step < steps; ++step) {
      const sliceStart = (dir === 1 || direction === 'backward' ? steps - step - 1 : step);
      const currentInput = squeeze(slice(input, [sliceStart, 0, 0], [1, batchSize, inputSize]));
      currentHidden = gruCell(
          currentInput, currentWeight[dir], currentRecurrentWeight[dir],
          currentHidden, hiddenSize, {bias: currentBias[dir],
            recurrentBias: currentRecurrentBias[dir], resetAfter, layout, activations});

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
  }

  if (returnSequence) {
    // outputSequence: [steps, numDirections, batchSize, hiddenSize]
    let outputSequence = null;
    if (direction === 'forward') {
      outputSequence = forwardSequence;
    } else if (direction === 'backward') {
      outputSequence = backwardSequence;
    } else if (direction === 'both') {
      // Concat along axis 1 (numDirections dimension)
      outputSequence = concat([forwardSequence, backwardSequence], 1);
    }
    return [outputHidden, outputSequence];
  } else {
    return [outputHidden];
  }
}
