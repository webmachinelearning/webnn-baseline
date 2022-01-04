'use strict';

/**
 * Compute the number of elements given a shape.
 * @param {Array} shape
 * @return {Number}
 */
export function sizeOfShape(shape) {
  return shape.reduce(
      (accumulator, currentValue) => accumulator * currentValue, 1);
}

/**
 * Tensor: the multidimensional array.
 */
export class Tensor {
  /**
   * Construct a Tensor object
   * @param {Array} shape
   * @param {Array} [data]
   */
  constructor(shape, data = undefined) {
    const size = sizeOfShape(shape);
    if (data !== undefined) {
      if (size !== data.length) {
        throw new Error(`The length of data ${data.length} is invalid, expected ${size}.`);
      }
      // Copy the data.
      this.data = data.slice();
    } else {
      this.data = new Array(size).fill(0);
    }
    // Copy the shape.
    this.shape = shape.slice();
    // Calculate the strides.
    this.strides = new Array(this.rank);
    this.strides[this.rank - 1] = 1;
    for (let i = this.rank - 2; i >= 0; --i) {
      this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
    }
  }

  get rank() {
    return this.shape.length;
  }

  get size() {
    return this.data.length;
  }

  /**
   * Get index in the flat array given the location.
   * @param {Array} location
   * @return {Number}
   */
  indexFromLocation(location) {
    if (location.length !== this.rank) {
      throw new Error(`The location length ${location.length} is not equal to rank ${this.rank}.`);
    }
    let index = 0;
    for (let i = 0; i < this.rank; ++i) {
      if (location[i] >= this.shape[i]) {
        throw new Error(`The location value ${location[i]} at axis ${i} is invalid.`);
      }
      index += this.strides[i] * location[i];
    }
    return index;
  }

  /**
   * Get location from the index of the flat array.
   * @param {Number} index
   * @return {Array}
   */
  locationFromIndex(index) {
    if (index >= this.size) {
      throw new Error('The index is invalid.');
    }
    const location = new Array(this.rank);
    for (let i = 0; i < location.length; ++i) {
      location[i] = Math.floor(index / this.strides[i]);
      index -= location[i] * this.strides[i];
    }
    return location;
  }

  /**
   * Set value given the location.
   * @param {Array} location
   * @param {Number} value
   */
  setValueByLocation(location, value) {
    this.data[this.indexFromLocation(location)] = value;
  }

  /**
   * Get value given the location.
   * @param {Array} location
   * @return {Number}
   */
  getValueByLocation(location) {
    return this.data[this.indexFromLocation(location)];
  }

  /**
   * Set value given the index.
   * @param {Number} index
   * @param {Number} value
   */
  setValueByIndex(index, value) {
    if (index >= this.size) {
      throw new Error('The index is invalid.');
    }
    this.data[index] = value;
  }

  /**
   * Get value given the index.
   * @param {Number} index
   * @return {Number}
   */
  getValueByIndex(index) {
    if (index >= this.size) {
      throw new Error('The index is invalid.');
    }
    return this.data[index];
  }
}
