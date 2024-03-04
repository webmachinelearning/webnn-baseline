'use strict';

import {Tensor} from '../src/lib/tensor.js';

/**
 * Cast each element in the input tensor to the target data type.
 * @param {Tensor} input
 * @return {Tensor}
 */

export function cast(input, type) {
  let outputArray;
  switch (type) {
    case 'int8':
      outputArray = new Int8Array(input.data);
      break;
    case 'uint8':
      outputArray = new Uint8Array(input.data);
      break;
    case 'int32':
      outputArray = new Int32Array(input.data);
      break;
    case 'uint32':
      outputArray = new Uint32Array(input.data);
      break;
    case 'int64':
      outputArray = new BigInt64Array(Array.from(input.data, (num) => BigInt(Math.trunc(num))));
      break;
    case 'float32':
      outputArray = new Float32Array(input.data);
      break;
    case 'float64':
      outputArray = new Float64Array(input.data);
      break;
    case 'float16':
      // TODO: https://github.com/webmachinelearning/webnn-baseline/issues/66
      throw new Error('Unsupported output type: float16');
    case 'uint64':
      // TODO: https://github.com/webmachinelearning/webnn-baseline/issues/67
      throw new Error('Unsupported output type: uint64');
    default:
      throw new Error('Unsupported output type: ' + type);
  }
  const output = new Tensor(input.shape, outputArray);
  return output;
}
