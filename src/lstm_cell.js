'use strict';

import {add, mul} from './binary.js';
import {matmul} from './matmul.js';
import {Scalar} from './lib/tensor.js';
import {sigmoid} from './sigmoid.js';
import {slice} from './slice.js';
import {tanh} from './tanh.js';
import {transpose} from './transpose.js';
import {validateLstmCellParams} from './lib/validate-input.js';

/**
 *A single time step of the Long Short-Term Memory [LSTM] recurrent network
 *using a cell state, an input, output, and forget gate to compute the cell
 *state and the hidden state of the next time step that rolls into the output
 *across the temporal sequence of the network.
 * @param {Tensor} input
 * @param {Tensor} weight
 * @param {Tensor} recurrentWeight
 * @param {Tensor} hiddenState
 * @param {Tensor} cellState
 * @param {Number} hiddenSize
 * @param {MLLstmCellOptions} options
 * @return {Tensor}
 */
export function lstmCell(input, weight, recurrentWeight, hiddenState, cellState, hiddenSize,
    {bias, recurrentBias, peepholeWeight,
      layout = 'iofg', activations = [sigmoid, tanh, tanh]}={}) {
  validateLstmCellParams(...arguments);
  const zero = new Scalar(0);
  const inputSize = input.shape[1];
  const starts = (layout === 'iofg') ? {i: 0, o: hiddenSize, f: 2 * hiddenSize, g: 3 * hiddenSize} :
                                       {i: 0, f: hiddenSize, g: 2 * hiddenSize, o: 3 * hiddenSize};
  const activation0 = activations[0];
  const activation1 = activations[1];
  const activation2 = activations[2];

  // input gate (i)
  const i = activation0(
      add(
          mul(
              cellState,
              (peepholeWeight ? slice(peepholeWeight, [0], [hiddenSize]) : zero),
          ),
          add(
              add(
                  (bias ? slice(bias, [starts.i], [hiddenSize]) : zero),
                  (recurrentBias ? slice(recurrentBias, [starts.i], [hiddenSize]) : zero),
              ),
              add(
                  matmul(
                      input,
                      transpose(slice(weight, [starts.i, 0], [hiddenSize, inputSize])),
                  ),
                  matmul(
                      hiddenState,
                      transpose(slice(recurrentWeight, [starts.i, 0], [hiddenSize, hiddenSize])),
                  ),
              ),
          ),
      ),
  );

  // forget gate (f)
  const f = activation0(
      add(
          mul(
              cellState,
              (peepholeWeight ? slice(peepholeWeight, [2 * hiddenSize], [hiddenSize]) : zero),
          ),
          add(
              add(
                  (bias ? slice(bias, [starts.f], [hiddenSize]) : zero),
                  (recurrentBias ? slice(recurrentBias, [starts.f], [hiddenSize]) : zero),
              ),
              add(
                  matmul(
                      input,
                      transpose(slice(weight, [starts.f, 0], [hiddenSize, inputSize])),
                  ),
                  matmul(
                      hiddenState,
                      transpose(
                          slice(recurrentWeight, [starts.f, 0], [hiddenSize, hiddenSize]),
                      ),
                  ),
              ),
          ),
      ),
  );

  // cell gate (g)
  const g = activation1(
      add(
          add(
              (bias ? slice(bias, [starts.g], [hiddenSize]) : zero),
              (recurrentBias ? slice(recurrentBias, [starts.g], [hiddenSize]) : zero),
          ),
          add(
              matmul(
                  input,
                  transpose(slice(weight, [starts.g, 0], [hiddenSize, inputSize])),
              ),
              matmul(
                  hiddenState,
                  transpose(slice(recurrentWeight, [starts.g, 0], [hiddenSize, hiddenSize])),
              ),
          ),
      ),
  );

  // output gate (o)
  const o = activation0(
      add(
          mul(
              cellState,
              (peepholeWeight ? slice(peepholeWeight, [hiddenSize], [hiddenSize]) : zero),
          ),
          add(
              add(
                  (bias ? slice(bias, [starts.o], [hiddenSize]) : zero),
                  (recurrentBias ? slice(recurrentBias, [starts.o], [hiddenSize]) : zero),
              ),
              add(
                  matmul(
                      input,
                      transpose(slice(weight, [starts.o, 0], [hiddenSize, inputSize])),
                  ),
                  matmul(
                      hiddenState,
                      transpose(slice(recurrentWeight, [starts.o, 0], [hiddenSize, hiddenSize])),
                  ),
              ),
          ),
      ),
  );

  // output cell state (ct)
  const cellStateNew = add(mul(f, cellState), mul(i, g));

  // output hidden state (ht)
  const hiddenStateNew = mul(o, activation2(cellStateNew));

  return [hiddenStateNew, cellStateNew];
}
