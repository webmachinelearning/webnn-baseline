'use strict';

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
    const size = shape.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
    if (data !== undefined) {
      if (size !== data.length) {
        throw new Error('The length of array is invalid.');
      }
      this.data = data;
    } else {
      this.data = new Array(size).fill(0);
    }
    this.shape = shape;
    this.rank = shape.length;

    if (this.rank < 2) {
      this.strides = [];
    } else {
      this.strides = new Array(this.rank - 1);
      this.strides[this.rank - 2] = this.shape[this.rank - 1];
      for (let i = this.rank - 3; i >= 0; --i) {
        this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
      }
    }
  }

  /**
   * Get index in the flat array given the location.
   * @param {Array} location
   * @return {Number}
   */
  indexFromLocation(location) {
    if (location.length !== this.rank) {
      throw new Error('The location is invalid.');
    }
    let index = 0;
    for (let i = 0; i < location.length - 1; ++i) {
      index += this.strides[i] * location[i];
    }
    index += location[location.length - 1];
    return index;
  }

  locationFromIndex(index) {
    const location = new Array(this.rank);
    for (let i = 0; i < location.length - 1; ++i) {
      location[i] = Math.floor(index / this.strides[i]);
      index -= location[i] * this.strides[i];
    }
    location[location.length - 1] = index;
    return location;
  }

  /**
   * Set value given the location.
   * @param {Array} location
   * @param {Number} value
   */
  setValue(location, value) {
    this.data[this.indexFromLocation(location)] = value;
  }

  /**
   * Get value given the location.
   * @param {Array} location
   * @return {Number}
   */
  getValue(location) {
    return this.data[this.indexFromLocation(location)];
  }
}
