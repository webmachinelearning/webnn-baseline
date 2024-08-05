'use strict';

import {gelu} from '../src/gelu.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test gelu', function() {
  function testGelu(inputShape, inputValue, expected) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = gelu(inputTensor);
    utils.checkValue(outputTensor, expected);
  }

  it('gelu', function() {
    // Refer to ONNX gelu_default test:
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#gelu
    testGelu([3], [-1, 0, 1], [-0.15865526383236372, 0, 0.8413447361676363]);
    testGelu([1, 1, 1, 3], [-1, 0, 1], [-0.15865526383236372, 0, 0.8413447361676363]);
  });
});
