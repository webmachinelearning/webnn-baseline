'use strict';

import {Scalar, Tensor} from './lib/tensor.js';
import {add, div, mul, sub} from '../src/binary.js';

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

export const abs = (input) => unary(input, Math.abs);
export const ceil = (input) => unary(input, Math.ceil);
export const cos = (input) => unary(input, Math.cos);
export const exp = (input) => unary(input, Math.exp);
export const floor = (input) => unary(input, Math.floor);
export const log = (input) => unary(input, Math.log);
export const neg = (input) => unary(input, (x) => -1 * x);
export const sin = (input) => unary(input, Math.sin);
export const tan = (input) => unary(input, Math.tan);
export const copy = (input) => unary(input, (x) => x);
export const reciprocal = (input) => unary(input, (x) => 1 / x);
export const sqrt = (input) => unary(input, Math.sqrt);
export const sign = (input) => unary(input, Math.sign);

export const erf = (input) => {
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
};
