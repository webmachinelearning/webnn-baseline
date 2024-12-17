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
 * Asserts that the distance between a and b whether is close enough to the given ULP distance.
 * @param {Number} a
 * @param {Number} b
 * @param {Number} nulp A BigInt value.
 * @param {String} message A message to report when the assertion fails
 * @return {Boolean} A boolean value:
 *     true: The distance between a and b is close enough to the given ULP distance.
 *     false: The distance between a and b is far away from the given ULP distance.
 */
assert.isAlmostEqualUlp = function(a, b, nulp, message) {
  if (typeof(a) == 'number') {
    const aBitwise = getBitwise(a);
    const bBitwise = getBitwise(b);
    let distance = aBitwise - bBitwise;
    distance = distance >= 0 ? distance : -distance;
    return assert.isTrue(distance <= nulp, message);
  } else {
    let distance = a - b;
    distance = distance >= 0n ? distance : -distance;
    return assert.isTrue(distance <= nulp, message);
  }
};

export function checkValue(tensor, expected, nulp = 0) {
  assert.isTrue(tensor.size === expected.length);
  for (let i = 0; i < expected.length; ++i) {
    assert.isAlmostEqualUlp(tensor.getValueByIndex(i), expected[i], nulp,
        `${tensor.getValueByIndex(i)} is almost equal to ${expected[i]}`);
  }
}

export function checkValueByATOL(actual, expected, tolerance) {
  assert.isTrue(actual.size === expected.length);
  for (let i = 0; i < expected.length; ++i) {
    const actualValue = actual.getValueByIndex(i);
    assert.isTrue(actualValue >= expected[i] - tolerance && actualValue <= expected[i] + tolerance,
        `${actualValue} is almost equal to ${expected[i]} by ${tolerance} tolerance of ATOL` +
        'metrics');
  }
}

export function checkShape(tensor, expected) {
  assert.equal(tensor.rank, expected.length,
      `Tensor has expected rank ${expected.length}: ${tensor.rank}`);
  for (let i = 0; i < expected.length; ++i) {
    assert.equal(tensor.shape[i], expected[i],
        `Tensor line ${i} has expected length ${expected[i]}: ${tensor.shape[i]}`);
  }
}

export function bindTrailingArgs(fn, ...boundArgs) {
  return function(...args) {
    return fn(...args, ...boundArgs);
  };
}
