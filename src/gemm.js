'use strict';

import {add, mul} from './binary.js';
import {matmul} from './matmul.js';
import {Scalar} from './lib/tensor.js';
import {validateInput} from './lib/validate-input.js';
import {transpose} from './transpose.js';

/**
 * Calculate the general matrix multiplication of the Basic Linear Algebra Subprograms.
 * The calculation follows the expression alpha * A * B + beta * C
 * @param {Tensor} a
 * @param {Tensor} b
 * @param {MLGemmOptions} options
 * @return {Tensor}
 */
export function gemm(a, b, {c = new Scalar(0.0),
  alpha: fAlpha = 1.0,
  beta: fBeta = 1.0,
  aTranspose = false,
  bTranspose = false,
} = {}) {
  validateInput('gemm', arguments);
  const alpha = new Scalar(fAlpha);
  const beta = new Scalar(fBeta);
  if (aTranspose) {
    a = transpose(a);
  }

  if (bTranspose) {
    b = transpose(b);
  }

  let output = matmul(mul(a, alpha), b);
  if (c) {
    output = add(output, mul(c, beta));
  }

  return output;
}
