'use strict';

import {concat} from './concat.js';
import {gruCell} from './gru_cell.js';
import {reshape} from './reshape.js';
import {sizeOfShape, Tensor} from './tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
import {squeeze} from './squeeze.js';
import {tanh} from './tanh.js';

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
export function gru(input, weight, recurrentWeight, steps, hiddenSize, options = {}) {
  const bias = options.bias;
  const recurrentBias = options.recurrentBias;
  const initialHiddenState = options.initialHiddenState;
  const resetAfter = options.resetAfter !== undefined ? options.resetAfter : true;
  const returnSequence = options.returnSequence !== undefined ? options.returnSequence : false;
  const direction = options.direction !== undefined ? options.direction : 'forward';
  const layout = options.layout !== undefined ? options.layout : 'zrn';
  const activations = options.activations ? options.activations : [sigmoid, tanh];

  if (!Number.isInteger(steps) || steps <= 0) {
    throw new Error(`The steps ${steps} is invalid.`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
  }
  if (input.rank !== 3) {
    throw new Error(`The input (rank ${input.rank}) is not a 3-D tensor.`);
  }
  if (input.shape[0] !== steps) {
    throw new Error(`The input.shape[0] ${input.shape[0]} is not equal to steps ${steps}.`);
  }
  const batchSize = input.shape[1];
  const inputSize = input.shape[2];
  if (direction !== 'forward' && direction !== 'backward' && direction !== 'both') {
    throw new Error(`The direction ${direction} is invalid.`);
  }
  const numDirections = (direction === 'both' ? 2 : 1);
  if (weight.rank !== 3) {
    throw new Error(`The weight (rank ${weight.rank}) is not a 3-D tensor.`);
  }
  if (weight.shape[0] !== numDirections || weight.shape[1] !== 3 * hiddenSize ||
      weight.shape[2] !== inputSize) {
    throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]},
        ${weight.shape[2]}] is invalid.`);
  }
  if (recurrentWeight.rank !== 3) {
    throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 3-D tensor.`);
  }
  if (recurrentWeight.shape[0] !== numDirections || recurrentWeight.shape[1] !== 3 * hiddenSize ||
      recurrentWeight.shape[2] !== hiddenSize) {
    throw new Error(`The shape of recurrentWeight ` +
        `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}, ` +
        `${recurrentWeight.shape[2]}] is invalid.`);
  }
  if (bias) {
    if (bias.rank !== 2) {
      throw new Error(`The bias (rank ${bias.rank}) is not a 2-D tensor.`);
    }
    if (bias.shape[0] !== numDirections || bias.shape[1] !== 3 * hiddenSize) {
      throw new Error(`The shape of bias [${bias.shape[0]}, ${bias.shape[1]}] is invalid.`);
    }
  }
  if (recurrentBias) {
    if (recurrentBias.rank !== 2) {
      throw new Error(`The recurrentBias (rank ${recurrentBias.rank}) is not a 2-D tensor.`);
    }
    if (recurrentBias.shape[0] !== numDirections || recurrentBias.shape[1] !== 3 * hiddenSize) {
      throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]},
          ${recurrentBias.shape[1]}] is invalid.`);
    }
  }
  let hiddenState;
  if (initialHiddenState) {
    if (initialHiddenState.rank !== 3) {
      throw new Error(
          `The initialHiddenState (rank ${initialHiddenState.rank}) is not a 3-D tensor.`);
    }
    if (initialHiddenState.shape[0] !== numDirections ||
        initialHiddenState.shape[1] !== batchSize ||
        initialHiddenState.shape[2] !== hiddenSize) {
      throw new Error(`The shape of initialHiddenState [${initialHiddenState.shape[0]},
          ${initialHiddenState.shape[1]}, ${initialHiddenState.shape[2]}] is invalid.`);
    }
    hiddenState = initialHiddenState;
  } else {
    const initialHiddenStateShape = [numDirections, batchSize, hiddenSize];
    hiddenState = new Tensor(
        initialHiddenStateShape, new Array(sizeOfShape(initialHiddenStateShape)).fill(0));
  }
  if (layout !== 'zrn' && layout !== 'rzn') {
    throw new Error(`The layout ${layout} is invalid.`);
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
