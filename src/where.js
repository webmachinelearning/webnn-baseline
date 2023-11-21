'use strict';

import {broadcast, getBroadcastShape} from './lib/broadcast.js';
import {Tensor, sizeOfShape} from './lib/tensor.js';

/**
 * Select the values from the input or the other tensor depending on
 * the corresponding Boolean values of the condition tensor.
 * @param {Tensor} condition
 * @param {Tensor} inputA
 * @param {Tensor} inputB
 * @return {Tensor}
 */
export function where(condition, inputA, inputB) {
  const tempShape1 = getBroadcastShape(inputA.shape, inputB.shape);
  const tempShape2 = getBroadcastShape(condition.shape, inputB.shape);
  const outputShape = getBroadcastShape(tempShape1, tempShape2);
  const inputABroadcast = broadcast(inputA, outputShape);
  const inputBBroadcast = broadcast(inputB, outputShape);
  const conditionBroadcast = broadcast(condition, outputShape);
  const outputSize = sizeOfShape(outputShape);
  const output = new Tensor(outputShape);
  for (let i = 0; i < outputSize; ++i) {
    if (conditionBroadcast.getValueByIndex(i) === 1) {
      output.setValueByIndex(i, inputABroadcast.getValueByIndex(i));
    } else {
      output.setValueByIndex(i, inputBBroadcast.getValueByIndex(i));
    }
  }
  return output;
}
