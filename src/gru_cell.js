'use strict';

import {add, mul, sub} from './binary.js';
import {matmul} from './matmul.js';
import {Scalar} from './lib/tensor.js';
import {sigmoid} from './lib/sigmoid.js';
import {slice} from './slice.js';
import {tanh} from './tanh.js';
import {transpose} from './transpose.js';

/**
 * A single time step of the Gated Recurrent Unit [GRU] recurrent network using an update gate
 * and a reset gate to compute the hidden state that rolls into the output across the temporal
 * sequence of a recurrent network.
 * @param {Tensor} input
 * @param {Tensor} weight
 * @param {Tensor} recurrentWeight
 * @param {Tensor} hiddenState
 * @param {Number} hiddenSize
 * @param {MLGruCellOptions} options
 * @return {Tensor}
 */
export function gruCell(input, weight, recurrentWeight, hiddenState, hiddenSize, options = {}) {
  const bias = options.bias;
  const recurrentBias = options.recurrentBias;
  const resetAfter = options.resetAfter !== undefined ? options.resetAfter : true;
  const layout = options.layout !== undefined ? options.layout : 'zrn';
  const activations = options.activations ? options.activations : [sigmoid, tanh];

  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
  }
  if (input.rank !== 2) {
    throw new Error(`The input (rank ${input.rank}) is not a 2-D tensor.`);
  }
  const batchSize = input.shape[0];
  const inputSize = input.shape[1];
  if (weight.rank !== 2) {
    throw new Error(`The weight (rank ${weight.rank}) is not a 2-D tensor.`);
  }
  if (weight.shape[0] !== 3 * hiddenSize || weight.shape[1] !== inputSize) {
    throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]}] is invalid.`);
  }
  if (recurrentWeight.rank !== 2) {
    throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 2-D tensor.`);
  }
  if (recurrentWeight.shape[0] !== 3 * hiddenSize || recurrentWeight.shape[1] !== hiddenSize) {
    throw new Error(`The shape of recurrentWeight ` +
      `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}] is invalid.`);
  }
  if (hiddenState.rank !== 2) {
    throw new Error(`The hiddenState (rank ${hiddenState.rank}) is not a 2-D tensor.`);
  }
  if (hiddenState.shape[0] !== batchSize || hiddenState.shape[1] !== hiddenSize) {
    throw new Error(`The shape of hiddenState
      [${hiddenState.shape[0]}, ${hiddenState.shape[1]}] is invalid.`);
  }
  if (bias) {
    if (bias.rank !== 1) {
      throw new Error(`The bias (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (bias.shape[0] !== 3 * hiddenSize) {
      throw new Error(`The shape of bias [${bias.shape[0]}] is invalid.`);
    }
  }
  if (recurrentBias) {
    if (recurrentBias.rank !== 1) {
      throw new Error(`The recurrentBias (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (recurrentBias.shape[0] !== 3 * hiddenSize) {
      throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]}] is invalid.`);
    }
  }
  if (layout !== 'zrn' && layout !== 'rzn') {
    throw new Error(`The layout ${layout} is invalid.`);
  }

  const one = new Scalar(1);
  const zero = new Scalar(0);
  const starts = layout === 'zrn' ? {z: 0, r: hiddenSize, n: 2 * hiddenSize} :
    {r: 0, z: hiddenSize, n: 2 * hiddenSize};
  const activation0 = activations[0];
  const activation1 = activations[1];
  // update gate
  const z = activation0(
      add(
          add(
              (bias ? slice(bias, [starts.z], [hiddenSize]) : zero),
              (recurrentBias ? slice(recurrentBias, [starts.z], [hiddenSize]) :zero)),
          add(
              matmul(input, transpose(slice(weight, [starts.z, 0], [hiddenSize, -1]))),
              matmul(
                  hiddenState,
                  transpose(slice(recurrentWeight, [starts.z, 0], [hiddenSize, -1]))))));
  // reset gate
  const r = activation0(
      add(
          add(
              (bias ? slice(bias, [starts.r], [hiddenSize]) : zero),
              (recurrentBias ? slice(recurrentBias, [starts.r], [hiddenSize]) : zero)),
          add(
              matmul(input, transpose(slice(weight, [starts.r, 0], [hiddenSize, -1]))),
              matmul(
                  hiddenState,
                  transpose(slice(recurrentWeight, [starts.r, 0], [hiddenSize, -1]))))));
  // new gate
  let n;
  if (resetAfter) {
    n = activation1(
        add(
            (bias ? slice(bias, [starts.n], [hiddenSize]) : zero),
            add(
                matmul(input, transpose(slice(weight, [starts.n, 0], [hiddenSize, -1]))),
                mul(
                    r,
                    add(
                        (recurrentBias ? slice(recurrentBias, [starts.n], [hiddenSize]) : zero),
                        matmul(
                            hiddenState,
                            transpose(
                                slice(recurrentWeight, [starts.n, 0], [hiddenSize, -1]))))))));
  } else {
    n = activation1(
        add(
            add(
                (bias ? slice(bias, [starts.n], [hiddenSize]) : zero),
                (recurrentBias ? slice(recurrentBias, [starts.n], [hiddenSize]) : zero)),
            add(
                matmul(
                    input,
                    transpose(slice(weight, [starts.n, 0], [hiddenSize, -1]))),
                matmul(
                    mul(r, hiddenState),
                    transpose(slice(recurrentWeight, [starts.n, 0], [hiddenSize, -1]))))));
  }
  // compute the new hidden state
  return add(mul(z, hiddenState), mul(n, sub(one, z)));
}

