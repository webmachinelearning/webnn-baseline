import {Tensor} from './tensor.js';
import {expand} from '../expand.js';
import {reshape} from '../reshape.js';

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

export function blockwiseExpand(input, outputShape) {
  // Given the original input and a desired output shape, this expands each axis
  // by repeating the block the number of times per that axis. Though, backend
  // implementations might have much more efficient upsampling operators that
  // can accept multiple dimensions to upsample all dimensions at once by
  // integer multiples (like tile) using nearest neighbor resampling:
  // output = resample(scale, {sizes: input.shape})

  let output = input;

  for (let axis = 0; axis < input.shape.length; ++axis) {
    const oldShape = output.shape;
    const oldDimensionLength = oldShape[axis];
    const newDimensionLength = outputShape[axis];

    if (newDimensionLength != oldDimensionLength) {
      // Since tile/expand can only accept repetitions of entire dimension
      // slices (not repeating individual elements along an axis), temporarily
      // reshape the tensor to enable them to broadcast the elements up to the
      // full block size, utilizing an inserted dimension of size 1.
      const elementRepeatCount = newDimensionLength / oldDimensionLength;
      const flattenedShape = getFlattenedShapeAroundAxis(oldShape, axis);
      const unexpandedShape =
        [flattenedShape[0], flattenedShape[1], 1, flattenedShape[2]];
      const expandedShape = [
        flattenedShape[0],
        flattenedShape[1],
        elementRepeatCount,
        flattenedShape[2],
      ];
      const reshapedInput = reshape(output, unexpandedShape);
      output = expand(reshapedInput, expandedShape);

      const newShape = [...oldShape];
      newShape[axis] = newDimensionLength;
      output = reshape(output, newShape);
    }
  }

  return output;
}

// Compute the flattened shape before and after the given axis, yielding a
// 3-element list: e.g.
// - inputShape = [2,3,4,5,6] with axis = 2 yields shape [6,4,30].
// - inputShape = [4] with axis = 0 yields shape [1,4,1].
function getFlattenedShapeAroundAxis(inputShape, axis) {
  axis = Math.max(Math.min(axis, inputShape.length - 1), 0);
  const shapeBefore = inputShape.slice(0, axis);
  const shapeAfter = inputShape.slice(axis + 1, inputShape.length);
  const countBefore = shapeBefore.reduce((a, b) => a * b, 1);
  const countAfter = shapeAfter.reduce((a, b) => a * b, 1);
  return [countBefore, inputShape[axis], countAfter];
}
