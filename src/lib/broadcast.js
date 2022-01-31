import {Tensor} from './tensor.js';

/**
 * Broadcast a Tensor to a compatible shape NumPy-style.
 * @param {Tensor} input
 * @param {Array} newShape
 * @return {Tensor}
 */
export function broadcast(input, newShape) {
  const newRank = newShape.length;
  if (newRank < input.rank) {
    throw new Error(`The rank of new shape ${newRank} is invalid.`);
  }
  const broadcastAxes = new Array(input.rank).fill(false);
  for (let i = 0; i < input.rank; ++i) {
    const newAxis = newRank - i - 1;
    const axis = input.rank - i - 1;
    if (input.shape[axis] === 1 && newShape[newAxis] !== 1) {
      broadcastAxes[axis] = true;
    } else if (input.shape[axis] !== newShape[newAxis]) {
      throw new Error(`The size of new shape at axis ${newAxis} is invalid.`);
    }
  }
  const output = new Tensor(newShape);
  for (let index = 0; index < output.size; ++index) {
    const location = output.locationFromIndex(index);
    const inputLocation = location.slice(-input.rank);
    for (let axis = 0; axis < input.rank; ++axis) {
      if (broadcastAxes[axis] === true) {
        inputLocation[axis] = 0;
      }
    }
    const inputValue = input.getValueByLocation(inputLocation);
    output.setValueByIndex(index, inputValue);
  }
  return output;
}

/**
 * Get broadcast shape of given two input shapes, throw error if they're incompatible.
 * @param {Array} shapeA
 * @param {Array} shapeB
 * @return {Array}
 */
export function getBroadcastShape(shapeA, shapeB) {
  // According to General Broadcasting Rules on
  //   https://numpy.org/doc/stable/user/basics.broadcasting.html.
  const outShape = [];
  const lenA = shapeA.length;
  const lenB = shapeB.length;
  const outlen = Math.max(lenA, lenB);
  for (let i = 0; i < outlen; ++i) {
    let a = shapeA[lenA - i - 1];
    if (a === undefined) {
      a = 1;
    }
    let b = shapeB[lenB - i - 1];
    if (b === undefined) {
      b = 1;
    }
    if (a === 1) {
      outShape.unshift(b);
    } else if (b === 1) {
      outShape.unshift(a);
    } else if (a !== b) {
      throw new Error(`Shapes [${shapeA}] and [${shapeB}] are incompatible.`);
    } else {
      outShape.unshift(a);
    }
  }
  return outShape;
}
