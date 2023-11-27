'use strict';

import {broadcast, getBroadcastShape} from './lib/broadcast.js';
import {Tensor, sizeOfShape, Scalar} from './lib/tensor.js';

/**
 * Select the values from the input or the other tensor depending on
 * the corresponding Boolean values of the condition tensor.
 * @param {Tensor} condition
 * @param {Tensor} trueValues
 * @param {Tensor} falseValues
 * @return {Tensor}
 */
export function where(condition, trueValues, falseValues) {
  const trueValuesReshape = trueValues.shape.length === 0 ?
    new Scalar(trueValues.data) :trueValues;
  const falseValuesReshape = falseValues.shape.length === 0 ?
    new Scalar(falseValues.data) : falseValues;
  const valueShape = getBroadcastShape(trueValuesReshape.shape, falseValuesReshape.shape);
  const outputShape = getBroadcastShape(condition.shape, valueShape);
  const trueValuesReshapeBroadcast = broadcast(trueValuesReshape, outputShape);
  const falseValuesReshapeBroadcast = broadcast(falseValuesReshape, outputShape);
  const conditionBroadcast = broadcast(condition, outputShape);
  const outputSize = sizeOfShape(outputShape);
  const output = new Tensor(outputShape);
  for (let i = 0; i < outputSize; ++i) {
    const value = conditionBroadcast.getValueByIndex(i) === 0 ?
    falseValuesReshapeBroadcast.getValueByIndex(i) : trueValuesReshapeBroadcast.getValueByIndex(i);
    output.setValueByIndex(i, value);
  }
  return output;
}
