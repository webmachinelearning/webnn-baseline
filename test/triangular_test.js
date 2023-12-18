'use strict';

import {triangular} from '../src/triangular.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test triangular', function() {
  function testTriangular(input, expected, options = {}) {
    const x = new Tensor(input.shape, input.data);
    const y = triangular(x, options);
    utils.checkShape(y, expected.shape);
    utils.checkValue(y, expected.data);
  }

  it('triangular default', function() {
    testTriangular(
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            0, 4, 8,
            0, 0, 3,
          ],
        },
    );
  });

  it('triangular diagonal=1', function() {
    testTriangular(
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          shape: [3, 3],
          data: [
            0, 1, 2,
            0, 0, 8,
            0, 0, 0,
          ],
        },
        {
          diagonal: 1,
        },
    );
  });

  it('triangular diagonal=-1', function() {
    testTriangular(
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            0, 6, 3,
          ],
        },
        {
          diagonal: -1,
        },
    );
  });

  it('triangular upper=false', function() {
    testTriangular(
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          shape: [3, 3],
          data: [
            7, 0, 0,
            9, 4, 0,
            2, 6, 3,
          ],
        },
        {
          upper: false,
        },
    );
  });

  it('triangular upper=false diagonal=1', function() {
    testTriangular(
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          shape: [3, 3],
          data: [
            7, 1, 0,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          upper: false,
          diagonal: 1,
        },
    );
  });

  it('triangular upper=false diagonal=-1', function() {
    testTriangular(
        {
          shape: [3, 3],
          data: [
            7, 1, 2,
            9, 4, 8,
            2, 6, 3,
          ],
        },
        {
          shape: [3, 3],
          data: [
            0, 0, 0,
            9, 0, 0,
            2, 6, 0,
          ],
        },
        {
          upper: false,
          diagonal: -1,
        },
    );
  });
});
