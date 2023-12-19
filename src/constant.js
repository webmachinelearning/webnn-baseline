'use strict';

import {cast} from './cast.js';
import {Tensor, sizeOfShape} from '../src/lib/tensor.js';

/**
 * Create a constant array of the specified data type and
 * shape that contains the data as specified by the range.
 * @param {Array} outputShape
 * @return {Tensor}
 */

export function constant(start, step, outputShape, type = 'float32') {
  const outputElementCount = sizeOfShape(outputShape);
  const resultArray = [];
  for (let i = 0; i < outputElementCount; i++) {
    resultArray.push(start + i * step);
  }
  const resultToTensor = new Tensor([resultArray.length], resultArray);

  const transformTensorType = cast(resultToTensor, type);
  const output = new Tensor(outputShape, transformTensorType.data);
  return output;
}
