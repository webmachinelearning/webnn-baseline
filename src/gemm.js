'use strict';

import {add, mul} from './binary.js';
import {matmul} from './matmul.js';
import {Scalar} from './lib/tensor.js';
import {transpose} from './transpose.js';

/**
 * Calculate the general matrix multiplication of the Basic Linear Algebra Subprograms.
 * The calculation follows the expression alpha * A * B + beta * C
 * @param {Tensor} a
 * @param {Tensor} b
 * @param {MLGemmOptions} options
 * @return {Tensor}
 */
export function gemm(a, b, options = {}) {
  if (a.rank !== 2) {
    throw new Error('The input a is not a 2-D tensor.');
  }
  if (b.rank !== 2) {
    throw new Error('The input b is not a 2-D tensor.');
  }
  const c = options.c ? options.c : undefined;
  const alpha = new Scalar(options.alpha ? options.alpha : 1.0);
  const beta = new Scalar(options.beta ? options.beta : 1.0);
  const aTranspose = options.aTranspose ? options.aTranspose : false;
  const bTranspose = options.bTranspose ? options.bTranspose : false;

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
