'use strict';

import {Tensor} from './lib/tensor.js';

/**
 * Compute the element-wise unary operation for input tensor.
 * @param {Tensor} input
 * @param {Function} unaryFunc
 * @return {Tensor}
 */
export function unary(input, unaryFunc) {
  const output = new Tensor(input.shape);
  for (let i = 0; i < input.size; ++i) {
    const x = input.getValueByIndex(i);
    const y = unaryFunc(x);
    output.setValueByIndex(i, y);
  }
  return output;
}

export function erfKernel(x) {
  // reference 1: https://en.wikipedia.org/wiki/Error_function
  // reference 2: https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Erf.ts
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 =-1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = Math.sign(x);
  const v = Math.abs(x);
  const t = 1.0 / (1.0 + p * v);
  return sign *
      (1.0 -
          (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
          Math.exp(-v * v));
}

export const abs = (input) => unary(input, Math.abs);
export const ceil = (input) => unary(input, Math.ceil);
export const cos = (input) => unary(input, Math.cos);
export const exp = (input) => unary(input, Math.exp);
export const floor = (input) => unary(input, Math.floor);
export const log = (input) => unary(input, Math.log);
export const neg = (input) => unary(input, (x) => -1 * x);
export const sin = (input) => unary(input, Math.sin);
export const tan = (input) => unary(input, Math.tan);
export const identity = (input) => unary(input, (x) => x);
export const reciprocal = (input) => unary(input, (x) => 1 / x);
export const sqrt = (input) => unary(input, Math.sqrt);
export const erf = (input) => unary(input, erfKernel);
export const sign = (input) => unary(input, Math.sign);
