'use strict';

const assert = chai.assert;

/**
 * Get bitwise of the given value.
 * @param {Number} value
 * @return {Number} A 64-bit signed integer.
 */
function getBitwise(value) {
  const buffer = new ArrayBuffer(8);
  const int64Array = new BigInt64Array(buffer);
  int64Array[0] = value < 0 ? ~BigInt(0) : BigInt(0);
  const f64Array = new Float64Array(buffer);
  f64Array[0] = value;
  return int64Array[0];
}

/**
 * Check the distance between a and b whether is close enough to the given ULP distance.
 * @param {Number} a
 * @param {Number} b
 * @param {Number} nulp A BigInt value.
 * @return {Boolean} A boolean value:
 *     true: The distance between a and b is close enough to the given ULP distance.
 *     false: The distance between a and b is far away from the given ULP distance.
 */
function almostEqualUlp(a, b, nulp) {
  const aBitwise = getBitwise(a);
  const bBitwise = getBitwise(b);
  let distance = aBitwise - bBitwise;
  distance = distance >= 0 ? distance : -distance;
  return distance <= nulp;
}

export function checkValue(tensor, expected, nulp = 0n) {
  assert.isTrue(tensor.size === expected.length);
  for (let i = 0; i < expected.length; ++i) {
    assert.isTrue(almostEqualUlp(tensor.getValueByIndex(i), expected[i], nulp));
  }
}

export function checkShape(tensor, expected) {
  assert.equal(tensor.rank, expected.length);
  for (let i = 0; i < expected.length; ++i) {
    assert.equal(tensor.shape[i], expected[i]);
  }
}

export function bindTrailingArgs(fn, ...boundArgs) {
  return function(...args) {
    return fn(...args, ...boundArgs);
  };
}
