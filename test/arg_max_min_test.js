'use strict';

import {argMax, argMin} from '../src/arg_max_min.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';

describe('test argMax and argMin', function() {
  function testArgMaxMin(input, expected, func, options = {}) {
    const inputTensor = new Tensor(input.shape, input.value);
    const outputTensor = func(inputTensor, options);
    utils.checkShape(outputTensor, expected.shape);
    utils.checkValue(outputTensor, expected.value);
  }

  it('argMax default', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        {
          shape: [],
          value: [7],
        },
        argMax);
  });

  it('argMax axes=[]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        {
          shape: [3, 3],
          value: [
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
          ],
        },
        argMax,
        {
          axes: [],
        });
  });

  it('argMax scalar axes=[]', function() {
    testArgMaxMin(
        {
          shape: [],
          value: [3],
        },
        {
          shape: [],
          value: [0],
        },
        argMax,
        {
          axes: [],
        });
  });

  it('argMax scalar axes=[] no effect by both keepDimensions and selectLastIndex being true',
      function() {
        testArgMaxMin(
            {
              shape: [],
              value: [3],
            },
            {
              shape: [],
              value: [0],
            },
            argMax,
            {
              axes: [],
              keepDimensions: true,
              selectLastIndex: true,
            });
      });

  it('argMax axes=[0]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        {
          shape: [3],
          value: [1, 2, 1],
        },
        argMax,
        {
          axes: [0],
        });
  });

  it('argMax axes=[1]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        {
          shape: [3],
          value: [2, 2, 1],
        },
        argMax,
        {
          axes: [1],
        });
  });

  it('argMax 2D of shape [1, 3] axes=[1]', function() {
    testArgMaxMin(
        {
          shape: [1, 3],
          value: [1, 2, 3],
        },
        {
          shape: [1],
          value: [2],
        },
        argMax,
        {
          axes: [1],
        });
  });

  it('argMax axes=[0, 1]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        {
          shape: [],
          value: [7],
        },
        argMax,
        {
          axes: [0, 1],
        });
  });

  it('argMax axes=[1, 0]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 2,
          ],
        },
        {
          shape: [],
          value: [7],
        },
        argMax,
        {
          axes: [1, 0],
        });
  });

  it('argMax both keepDimensions and selectLastIndex being true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 5,
          ],
        },
        {
          shape: [1, 1],
          value: [8],
        },
        argMax,
        {
          keepDimensions: true,
          selectLastIndex: true,
        });
  });

  it('argMax axes=[1] keepDimensions=true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 5,
          ],
        },
        {
          shape: [3, 1],
          value: [2, 2, 1],
        },
        argMax,
        {
          axes: [1],
          keepDimensions: true,
        });
  });

  it('argMax axes=[1] both keepDimensions and selectLastIndex being true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 4,
            2, 5, 5,
          ],
        },
        {
          shape: [3, 1],
          value: [2, 2, 2],
        },
        argMax,
        {
          axes: [1],
          keepDimensions: true,
          selectLastIndex: true,
        });
  });

  it('argMax 3D axes=[1, 2]', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 2,
            5, 1, 5, 0,
            8, 7, 5, 6,
            6, 4, 9, 2,
            1, 9, 4, 3,
          ],
        },
        {
          shape: [2],
          value: [8, 6],
        },
        argMax,
        {
          axes: [1, 2],
        });
  });

  it('argMax 3D discontinuous axes=[0, 2]', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 9,
            5, 9, 5, 0,
            8, 7, 8, 6,
            9, 4, 5, 2,
            1, 9, 4, 3,
          ],
        },
        {
          shape: [3],
          value: [4, 3, 1],
        },
        argMax,
        {
          axes: [0, 2],
        });
  });

  it('argMax 3D discontinuous axes=[2, 0] selectLastIndex=true', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 9,
            5, 9, 5, 0,
            8, 7, 8, 6,
            9, 4, 5, 2,
            1, 9, 4, 3,
          ],
        },
        {
          shape: [3],
          value: [6, 4, 5],
        },
        argMax,
        {
          axes: [2, 0],
          selectLastIndex: true,
        });
  });

  it('argMax 3D axes=[2, 1] selectLastIndex=true', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 2,
            5, 1, 5, 0,
            8, 7, 5, 6,
            6, 4, 9, 2,
            1, 9, 4, 3,
          ],
        },
        {
          shape: [2],
          value: [10, 9],
        },
        argMax,
        {
          axes: [2, 1],
          selectLastIndex: true,
        });
  });

  it('argMax 3D of shape [1, 3, 4] axes=[1] keepDimensions=false', function() {
    testArgMaxMin(
        {
          shape: [1, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 3,
            5, 4, 5, 0,
          ],
        },
        {
          shape: [1, 4],
          value: [2, 1, 2, 0],
        },
        argMax,
        {
          axes: [1],
          keepDimensions: false,
        });
  });

  it('argMax 3D of shape [1, 3, 4] axes=[1] selectLastIndex=true', function() {
    testArgMaxMin(
        {
          shape: [1, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 4, 3,
            5, 4, 5, 0,
          ],
        },
        {
          shape: [1, 4],
          value: [2, 2, 2, 1],
        },
        argMax,
        {
          axes: [1],
          selectLastIndex: true,
        });
  });

  it('argMin default', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [],
          value: [4],
        },
        argMin);
  });

  it('argMin axes=[0]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [3],
          value: [0, 1, 1],
        },
        argMin,
        {
          axes: [0],
        });
  });

  it('argMin axes=[1]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [3],
          value: [0, 1, 0],
        },
        argMin,
        {
          axes: [1],
        });
  });

  it('argMin 2D of shape [1, 3] axes=[1]', function() {
    testArgMaxMin(
        {
          shape: [1, 3],
          value: [1, 2, 3],
        },
        {
          shape: [1],
          value: [0],
        },
        argMin,
        {
          axes: [1],
        });
  });

  it('argMin axes=[0, 1]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [],
          value: [4],
        },
        argMin,
        {
          axes: [0, 1],
        });
  });

  it('argMin axes=[1, 0]', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [],
          value: [4],
        },
        argMin,
        {
          axes: [1, 0],
        });
  });

  it('argMin both keepDimensions and selectLastIndex being true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [1, 1],
          value: [5],
        },
        argMin,
        {
          keepDimensions: true,
          selectLastIndex: true,
        });
  });

  it('argMin axes=[1] keepDimensions=true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [3, 1],
          value: [0, 1, 0],
        },
        argMin,
        {
          axes: [1],
          keepDimensions: true,
        });
  });

  it('argMin axes=[1] both keepDimensions and selectLastIndex being true', function() {
    testArgMaxMin(
        {
          shape: [3, 3],
          value: [
            1, 2, 3,
            3, 0, 0,
            2, 5, 2,
          ],
        },
        {
          shape: [3, 1],
          value: [0, 2, 2],
        },
        argMin,
        {
          axes: [1],
          keepDimensions: true,
          selectLastIndex: true,
        });
  });

  it('argMin 3D of shape [1, 3, 4] axes=[1] keepDimensions=false selectLastIndex=true', function() {
    testArgMaxMin(
        {
          shape: [1, 3, 4],
          value: [
            1, 2, 1, 0,
            3, 4, 2, 2,
            5, 0, 5, 0,
          ],
        },
        {
          shape: [1, 4],
          value: [0, 2, 0, 2],
        },
        argMin,
        {
          axes: [1],
          keepDimensions: false,
          selectLastIndex: true,
        });
  });

  it('argMin 3D discontinuous axes=[0, 2]', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 2, 9,
            5, 9, 5, 0,
            8, 7, 1, 6,
            9, 3, 5, 2,
            1, 9, 0, 4,
          ],
        },
        {
          shape: [3],
          value: [0, 2, 3],
        },
        argMin,
        {
          axes: [0, 2],
        });
  });

  it('argMin 3D discontinuous axes=[2, 0] selectLastIndex=true', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 3, 3,
            3, 4, 2, 9,
            5, 9, 5, 0,
            8, 7, 1, 6,
            9, 3, 5, 2,
            1, 9, 0, 4,
          ],
        },
        {
          shape: [3],
          value: [6, 7, 6],
        },
        argMin,
        {
          axes: [2, 0],
          selectLastIndex: true,
        });
  });

  it('argMin 3D axes=[1, 2]', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 1, 3,
            3, 4, 2, 2,
            5, 0, 5, 0,
            8, 7, 5, 5,
            2, 4, 9, 2,
            1, 2, 4, 1,
          ],
        },
        {
          shape: [2],
          value: [9, 8],
        },
        argMin,
        {
          axes: [1, 2],
        });
  });

  it('argMin 3D axes=[2, 1] selectLastIndex=true', function() {
    testArgMaxMin(
        {
          shape: [2, 3, 4],
          value: [
            1, 2, 1, 3,
            3, 4, 2, 2,
            5, 0, 5, 0,
            8, 7, 5, 5,
            2, 4, 9, 2,
            1, 2, 4, 1,
          ],
        },
        {
          shape: [2],
          value: [11, 11],
        },
        argMin,
        {
          axes: [2, 1],
          selectLastIndex: true,
        });
  });
});
