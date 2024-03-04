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

  it('triangular 2D of square shape [3, 3] default', function() {
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

  it('triangular 2D of shape [3, 4] default', function() {
    testTriangular(
        {
          shape: [3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            2, 6, 3, 5,
          ],
        },
        {
          shape: [3, 4],
          data: [
            7, 1, 2, 3,
            0, 4, 8, 4,
            0, 0, 3, 5,
          ],
        },
    );
  });

  it('triangular 3D default', function() {
    testTriangular(
        {
          shape: [2, 3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            2, 6, 3, 5,
            2, 6, 3, 5,
            9, 4, 8, 4,
            7, 1, 2, 3,
          ],
        },
        {
          shape: [2, 3, 4],
          data: [
            7, 1, 2, 3,
            0, 4, 8, 4,
            0, 0, 3, 5,
            2, 6, 3, 5,
            0, 4, 8, 4,
            0, 0, 2, 3,
          ],
        },
    );
  });

  it('triangular 4D default', function() {
    testTriangular(
        {
          shape: [2, 2, 3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            2, 6, 3, 5,
            2, 6, 3, 5,
            9, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            7, 1, 2, 3,
            9, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            9, 4, 8, 4,
          ],
        },
        {
          shape: [2, 2, 3, 4],
          data: [
            7, 1, 2, 3,
            0, 4, 8, 4,
            0, 0, 3, 5,
            2, 6, 3, 5,
            0, 4, 8, 4,
            0, 0, 2, 3,
            2, 6, 3, 5,
            0, 1, 2, 3,
            0, 0, 8, 4,
            7, 1, 2, 3,
            0, 6, 3, 5,
            0, 0, 8, 4,
          ],
        },
    );
  });

  it('triangular 2D diagonal=1', function() {
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

  it('triangular 3D diagonal=1', function() {
    testTriangular(
        {
          shape: [2, 3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            2, 6, 3, 5,
            2, 6, 3, 5,
            9, 4, 8, 4,
            7, 1, 2, 3,
          ],
        },
        {
          shape: [2, 3, 4],
          data: [
            0, 1, 2, 3,
            0, 0, 8, 4,
            0, 0, 0, 5,
            0, 6, 3, 5,
            0, 0, 8, 4,
            0, 0, 0, 3,
          ],
        },
        {
          diagonal: 1,
        },
    );
  });

  it('triangular 2D fully zero diagonal=4', function() {
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
            0, 0, 0,
            0, 0, 0,
          ],
        },
        {
          diagonal: 4,
        },
    );
  });

  it('triangular 2D diagonal=-1', function() {
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


  it('triangular 4D diagonal=-1', function() {
    testTriangular(
        {
          shape: [2, 2, 3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            2, 6, 3, 5,
            2, 6, 3, 5,
            9, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            7, 1, 2, 3,
            9, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            9, 4, 8, 4,
          ],
        },
        {
          shape: [2, 2, 3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            0, 6, 3, 5,
            2, 6, 3, 5,
            9, 4, 8, 4,
            0, 1, 2, 3,
            2, 6, 3, 5,
            7, 1, 2, 3,
            0, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            0, 4, 8, 4,
          ],
        },
        {
          diagonal: -1,
        },
    );
  });

  it('triangular 2D fully copied diagonal=-4', function() {
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
            2, 6, 3,
          ],
        },
        {
          diagonal: -4,
        },
    );
  });

  it('triangular 2D upper=false', function() {
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

  it('triangular 2D upper=false diagonal=1', function() {
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

  it('triangular 4D upper=false diagonal=1', function() {
    testTriangular(
        {
          shape: [2, 2, 3, 4],
          data: [
            7, 1, 2, 3,
            9, 4, 8, 4,
            2, 6, 3, 5,
            2, 6, 3, 5,
            9, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            7, 1, 2, 3,
            9, 4, 8, 4,
            7, 1, 2, 3,
            2, 6, 3, 5,
            9, 4, 8, 4,
          ],
        },
        {
          shape: [2, 2, 3, 4],
          data: [
            7, 1, 0, 0,
            9, 4, 8, 0,
            2, 6, 3, 5,
            2, 6, 0, 0,
            9, 4, 8, 0,
            7, 1, 2, 3,
            2, 6, 0, 0,
            7, 1, 2, 0,
            9, 4, 8, 4,
            7, 1, 0, 0,
            2, 6, 3, 0,
            9, 4, 8, 4,
          ],
        },
        {
          upper: false,
          diagonal: 1,
        },
    );
  });

  it('triangular 2D fully copied upper=false diagonal=4 ', function() {
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
            2, 6, 3,
          ],
        },
        {
          upper: false,
          diagonal: 4,
        },
    );
  });

  it('triangular 2D upper=false diagonal=-1', function() {
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

  it('triangular 2D fully zero upper=false diagonal=-4', function() {
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
            0, 0, 0,
            0, 0, 0,
          ],
        },
        {
          upper: false,
          diagonal: -4,
        },
    );
  });
});
