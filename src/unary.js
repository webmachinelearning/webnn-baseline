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
