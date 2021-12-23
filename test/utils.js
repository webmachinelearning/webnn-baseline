'use strict';

const assert = chai.assert;

export class AccuracyCriterion {
  constructor(atol, rtol) {
    this.atol = atol;
    this.rtol = rtol;
  }
}

export const opFp32AccuracyCriteria =
    new AccuracyCriterion(1e-6, 5.0 * 1.1920928955078125e-7);

// The following 2 constants were used for converted tests from NNAPI CTS
export const ctsFp32RestrictAccuracyCriteria =
    new AccuracyCriterion(1e-5, 5.0 * 1.1920928955078125e-7);
export const ctsFp32RelaxedAccuracyCriteria =
    new AccuracyCriterion(5.0 * 0.0009765625, 5.0 * 0.0009765625);

// Refer to onnx/models
//   https://github.com/onnx/models/blob/master/workflow_scripts/ort_test_dir_utils.py#L239
// See details of modelFp32AccuracyCriteria setting:
//   https://github.com/webmachinelearning/webnn-polyfill/issues/55
export const modelFp32AccuracyCriteria = new AccuracyCriterion(1e-3, 1e-3);

export function almostEqual(a, b, criteria) {
  const delta = Math.abs(a - b);
  if (delta <= criteria.atol + criteria.rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

export function checkValue(
    output, expected, criteria = opFp32AccuracyCriteria) {
  assert.isTrue(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.isTrue(almostEqual(output[i], expected[i], criteria));
  }
}

export function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue, 1);
}

export function checkShape(shape, expected) {
  assert.equal(shape.length, expected.length);
  for (let i = 0; i < shape.length; ++i) {
    assert.equal(shape[i], expected[i]);
  }
}

export function computeExplicitPadding(
    inputSize, stride, filterSize, dilation = 1) {
  const outSize = Math.ceil(inputSize / stride);
  const effectiveFilterSize = (filterSize - 1) * dilation + 1;
  const neededInput = (outSize - 1) * stride + effectiveFilterSize;
  const totalPadding = Math.max(0, neededInput - inputSize);
  const paddingToBeginning = Math.floor(totalPadding / 2);
  const paddingToEnd = Math.floor((totalPadding + 1) / 2);
  return [paddingToBeginning, paddingToEnd];
}

export function bindTrailingArgs(fn, ...boundArgs) {
  return function(...args) {
    return fn(...args, ...boundArgs);
  };
}
