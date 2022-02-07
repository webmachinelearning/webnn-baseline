'use strict';

import {add, mul, sub} from './binary.js';
import {matmul} from './matmul.js';
import {Scalar} from './lib/tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
import {tanh} from './tanh.js';
import {transpose} from './transpose.js';
import {validateInput} from './lib/validate-input.js';

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
export function gruCell(input, weight, recurrentWeight, hiddenState, hiddenSize,
                        {bias, recurrentBias, resetAfter = true,
                         layout = 'zrn', activations = [sigmoid, tanh]} = {}) {
  validateInput("gruCell", arguments);

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

