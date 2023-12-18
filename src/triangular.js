'use strict';

import {Tensor, sizeOfShape} from './lib/tensor.js';
import {validateTriangularParams} from './lib/validate-input.js';

/**
 * Get retained boolean flag.
 * @param {Array} location
 * @param {Boolean} upper
 * @param {Number} diagonal
 * @return {Boolean}
 */
function isRetainedValue(location, upper, diagonal) {
  const [i, j] = location;
  return upper ? j >= i + diagonal : j <= i + diagonal;
}

/**
 * Given a 2-D tensor (matrix), return a 2-D tensor containing either the upper or lower triangular
 * part of the input tensor.
 * @param {Tensor} input
 * @param {MLTriangularOptions} [options]
 * @return {Tensor}
 */
export function triangular(input, {upper = true, diagonal = 0} = {}) {
  validateTriangularParams(...arguments);
  const shapeOutput = input.shape.slice();
  const output = new Tensor(shapeOutput);

  for (let outputIndex = 0; outputIndex < sizeOfShape(shapeOutput); ++outputIndex) {
    const outputLoc = output.locationFromIndex(outputIndex);
    const retainedFlag = isRetainedValue(outputLoc, upper, diagonal);
    const inputValue = retainedFlag ? input.getValueByLocation(outputLoc) : 0;
    output.setValueByLocation(outputLoc, inputValue);
  }

  return output;
}
