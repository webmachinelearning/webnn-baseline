'use strict';
import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateNotParams} from './lib/validate-input.js';
import {binary} from './binary.js';

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

export const equal = (inputA, inputB) => binary(inputA, inputB, (a, b) => (a == b ? 1 : 0));
export const greater = (inputA, inputB) => binary(inputA, inputB, (a, b) => (a > b ? 1 : 0));
export const greaterOrEqual =
    (inputA, inputB) => binary(inputA, inputB, (a, b) => (a >= b ? 1 : 0));
export const lesser = (inputA, inputB) => binary(inputA, inputB, (a, b) => (a < b ? 1 : 0));
export const lesserOrEqual =
    (inputA, inputB) => binary(inputA, inputB, (a, b) => (a <= b ? 1 : 0));
export const not = (input) => logicalNot(input);
