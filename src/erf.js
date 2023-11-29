'use strict';

import {Scalar} from '../src/lib/tensor.js';
import {abs, exp, neg, sign} from '../src/unary.js';
import {add, div, mul, sub} from '../src/binary.js';

/**
 * Performs the Gaussian error function (erf) on each element of
 * InputTensor, placing the result into the corresponding element of OutputTensor.
 * @param {Tensor} input
 * @return {Tensor}
 */

export function erf(input) {
  /**
  *reference 1:https://en.wikipedia.org/wiki/Error_function
  *reference 2:https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgl/src/kernels/Erf.ts
  */
  const a1 = new Scalar(0.254829592);
  const a2 = new Scalar(-0.284496736);
  const a3 = new Scalar(1.421413741);
  const a4 = new Scalar(-1.453152027);
  const a5 = new Scalar(1.061405429);
  const p = new Scalar(0.3275911);
  const ones = new Scalar(1);
  input = input.shape.length === 0 ? new Scalar(input.data) : input;
  const signInput = sign(input);
  input = abs(input);
  const t = div(ones, add(ones, mul(p, input)));
  const y = mul(add(mul(add(mul(add(mul(add(mul(a5, t), a4), t), a3), t), a2), t), a1), t);
  const result = mul(signInput, sub(ones, mul(y, exp(neg(mul(input, input))))));
  return result;
}

