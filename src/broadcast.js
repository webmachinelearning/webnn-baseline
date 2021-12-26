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
      broadcastAxes[newAxis] = true;
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
