'use strict';

import {concat} from './concat.js';
import {gruCell} from './gru_cell.js';
import {reshape} from './reshape.js';
import {sizeOfShape, Tensor} from './lib/tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
import {squeeze} from './squeeze.js';
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

  let hiddenState;
  if (initialHiddenState) {
    hiddenState = initialHiddenState;
  } else {
    const initialHiddenStateShape = [numDirections, batchSize, hiddenSize];
    hiddenState = new Tensor(
        initialHiddenStateShape, new Array(sizeOfShape(initialHiddenStateShape)).fill(0));
  }

  let sequence;
  const cellWeight = [];
  const cellRecurrentWeight = [];
  const cellBias = [];
  const cellRecurrentBias = [];

  for (let slot = 0; slot < numDirections; ++slot) {
    cellWeight.push(
        squeeze(slice(weight, [slot, 0, 0], [1, -1, -1]), [0]));
    cellRecurrentWeight.push(squeeze(
        slice(recurrentWeight, [slot, 0, 0], [1, -1, -1]), [0]));
    cellBias.push(
        bias ? (squeeze(slice(bias, [slot, 0], [1, -1]), [0])) :
                undefined);
    cellRecurrentBias.push(
        recurrentBias ? (squeeze(slice(recurrentBias, [slot, 0], [1, -1]), [0])) : undefined);
  }

  for (let step = 0; step < steps; ++step) {
    const cellHidden = [];
    let cellOutput;

    for (let slot = 0; slot < numDirections; ++slot) {
      cellHidden.push(squeeze(slice(hiddenState, [slot, 0, 0], [1, -1, -1]), [0]));
    }

    for (let slot = 0; slot < numDirections; ++slot) {
      const sliceStart = (slot === 1 || direction === 'backward' ? steps - step - 1 : step);
      const cellInput = squeeze(slice(input, [sliceStart, 0, 0], [1, -1, -1]), [0]);

      const result = reshape(
          gruCell(
              cellInput, cellWeight[slot], cellRecurrentWeight[slot],
              cellHidden[slot], hiddenSize, {bias: cellBias[slot],
                recurrentBias: cellRecurrentBias[slot], resetAfter, layout, activations}),
          [1, -1, hiddenSize]);

      cellOutput = (cellOutput ? concat([cellOutput, result], 0) : result);
    }

    hiddenState = cellOutput;

    if (returnSequence) {
      cellOutput = reshape(cellOutput, [1, numDirections, -1, hiddenSize]);
      sequence =
          (sequence ? concat([sequence, cellOutput], 0) : cellOutput);
    }
  }

  return [hiddenState, sequence];
}
