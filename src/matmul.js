
'use strict';

import {broadcast, getBroadcastShape} from './broadcast.js';
import {reshape} from './reshape.js';
import {sizeOfShape, Tensor} from './tensor.js';

/**
 * Compute the matrix product of two input tensors.
 * @param {Tensor} a
 * @param {Tensor} b
 * @return {Tensor}
 */
export function matmul(a, b) {
  let scalarOutput = false;
  if (a.rank === 1 && b.rank === 1) {
    scalarOutput = true;
  }
  if (a.rank === 1) {
    a = reshape(a, [1, a.shape[0]]);
  }
  const aRows = a.shape[a.rank - 2];
  const aCols = a.shape[a.rank - 1];

  if (b.rank === 1) {
    b = reshape(b, [b.shape[0], 1]);
  }
  const bRows = b.shape[b.rank - 2];
  const bCols = b.shape[b.rank - 1];

  if (aCols !== bRows) {
    throw new Error(
        `The columns (${aCols}) of input a is not equal to rows (${bRows}) of input b.`);
  }

  let cShape = [aRows, bCols];
  if (a.rank > 2 || b.rank > 2) {
    // Broadcast
    const aBatchDims = a.shape.slice(0, -2);
    const bBatchDims = b.shape.slice(0, -2);
    const outputBatchDims = getBroadcastShape(aBatchDims, bBatchDims);
    const aShape = outputBatchDims.concat(a.shape.slice(-2));
    a = broadcast(a, aShape);
    const bShape = outputBatchDims.concat(b.shape.slice(-2));
    b = broadcast(b, bShape);
    cShape = outputBatchDims.concat(cShape);
  }
  let c = new Tensor(cShape);

  for (let i = 0; i < sizeOfShape(cShape); ++i) {
    const cLoc = c.locationFromIndex(i);
    const m = cLoc[c.rank - 2];
    const n = cLoc[c.rank - 1];
    let cValue = 0;
    for (let k = 0; k < aCols; ++k) {
      let aLoc = cLoc.slice(0, -2);
      aLoc = aLoc.concat(m, k);
      let bLoc = cLoc.slice(0, -2);
      bLoc = bLoc.concat(k, n);
      const aValue = a.getValueByLocation(aLoc);
      const bValue = b.getValueByLocation(bLoc);
      cValue += aValue * bValue;
    }
    c.setValueByLocation(cLoc, cValue);
  }

  if (scalarOutput) {
    c = reshape(c, [1]);
  }

  return c;
}
