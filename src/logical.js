'use strict';

import {broadcast, getBroadcastShape} from './lib/broadcast.js';
import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateNotParams} from './lib/validate-input.js';

/**
 * Compute the element-wise logical operation of two input tensors.
 * @param {Tensor} inputA
 * @param {Tensor} inputB
 * @param {Function} logicalFunc
 * @return {Tensor}
 */
function logical(inputA, inputB, logicalFunc) {
  const outputShape = getBroadcastShape(inputA.shape, inputB.shape);
  const inputABroadcast = broadcast(inputA, outputShape);
  const inputBBroadcast = broadcast(inputB, outputShape);
  const outputSize = sizeOfShape(outputShape);
  const output = new Tensor(outputShape);
  for (let i = 0; i < outputSize; ++i) {
    const a = inputABroadcast.getValueByIndex(i);
    const b = inputBBroadcast.getValueByIndex(i);
    const c = logicalFunc(a, b) ? 1 : 0;
    output.setValueByIndex(i, c);
  }
  return output;
}

/**
 * Compute the element-wise logical not operation of input tensors.
 * @param {Tensor} input
 * @return {Tensor}
 */
function logicalNot(input) {
  validateNotParams(input);
  const outputShape = input.shape;
  const outputSize = sizeOfShape(outputShape);
  const output = new Tensor(outputShape);
  for (let i = 0; i < outputSize; ++i) {
    const a = input.getValueByIndex(i);
    const b = !a ? 1 : 0;
    output.setValueByIndex(i, b);
  }
  return output;
}

export const equal = (inputA, inputB) => logical(inputA, inputB, (a, b) => a == b);
export const greater = (inputA, inputB) => logical(inputA, inputB, (a, b) => a > b);
export const greaterOrEqual = (inputA, inputB) => logical(inputA, inputB, (a, b) => (a >= b));
export const lesser = (inputA, inputB) => logical(inputA, inputB, (a, b) => a < b);
export const lesserOrEqual = (inputA, inputB) => logical(inputA, inputB, (a, b) => (a <= b));
export const not = (input) => logicalNot(input);
