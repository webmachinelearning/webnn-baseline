'use strict';

import {broadcast, getBroadcastShape} from './lib/broadcast.js';
import {Tensor, sizeOfShape} from './lib/tensor.js';

/**
 * Compute the element-wise binary operation of two input tensors.
 * @param {Tensor} inputA
 * @param {Tensor} inputB
 * @param {Function} binaryFunc
 * @return {Tensor}
 */
export function binary(inputA, inputB, binaryFunc) {
  const outputShape = getBroadcastShape(inputA.shape, inputB.shape);
  const inputABroadcast = broadcast(inputA, outputShape);
  const inputBBroadcast = broadcast(inputB, outputShape);
  const outputSize = sizeOfShape(outputShape);
  const output = new Tensor(outputShape);
  for (let i = 0; i < outputSize; ++i) {
    const a = inputABroadcast.getValueByIndex(i);
    const b = inputBBroadcast.getValueByIndex(i);
    const c = binaryFunc(a, b);
    output.setValueByIndex(i, c);
  }
  return output;
}

export const add = (inputA, inputB) => binary(inputA, inputB, (a, b) => a + b);
export const sub = (inputA, inputB) => binary(inputA, inputB, (a, b) => a - b);
export const mul = (inputA, inputB) => binary(inputA, inputB, (a, b) => a * b);
export const div = (inputA, inputB) => binary(inputA, inputB, (a, b) => a / b);
export const max = (inputA, inputB) => binary(inputA, inputB, (a, b) => Math.max(a, b));
export const min = (inputA, inputB) => binary(inputA, inputB, (a, b) => Math.min(a, b));
export const pow = (inputA, inputB) => binary(inputA, inputB, (a, b) => Math.pow(a, b));
