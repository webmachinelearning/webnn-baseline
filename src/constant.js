'use strict';

import {cast} from './cast.js';
import {Tensor, sizeOfShape} from '../src/lib/tensor.js';
/**
 * Create a constant array of specified data type and shape,
 * which contains data incrementing by step.
 * @param {Number} start
 * @param {Number} step
 * @param {Array} outputShape
 * @param {string} type
 * @return {Tensor}
 */
export function constant(start, step, outputShape, type = 'float32') {
  const outputElementCount = sizeOfShape(outputShape);
  const data = [];
  for (let i = 0; i < outputElementCount; i++) {
    data.push(start + i * step);
  }
  const tensor = new Tensor(outputShape, data);
  return cast(tensor, type);
}
