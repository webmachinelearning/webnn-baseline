'use strict';

import * as reducers from '../src/reduce.js';
import {Tensor} from '../src/lib/tensor.js';

import * as utils from './utils.js';

describe('test reduce', function() {
  function testReduce(op, options, input, expected) {
    const x = new Tensor(input.shape, input.values);
    const y = reducers['reduce' + op](x, options);
    utils.checkShape(y, expected.shape);
    utils.checkValue(y, expected.values);
  }

  it('reduceMax default', function() {
    testReduce(
        'Max', {}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [], values: [600.]});
  });

  it('reduceMax default axes keep dims', function() {
    testReduce(
        'Max', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [1, 1, 1], values: [600.]});
  });

  it('reduceMax axes0 do not keep dims', function() {
    testReduce(
        'Max', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [2, 2], values: [500., 100., 600., 400.]});
  });

  it('reduceMax axes1 do not keep dims', function() {
    testReduce(
        'Max', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2], values: [200., 100., 300., 400., 600., 6.]});
  });

  it('reduceMax axes2 do not keep dims', function() {
    testReduce(
        'Max', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2], values: [100., 200., 300., 400., 500., 600.]});
  });

  it('reduceMax 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'Max',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [1., 100., 200., 2.],
        },
        {
          shape: [1, 2],
          values: [100., 200.],
        });
  });

  it('reduceMax axes0 keep dims', function() {
    testReduce(
        'Max', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [1, 2, 2], values: [500., 100., 600., 400.]});
  });

  it('reduceMax axes1 keep dims', function() {
    testReduce(
        'Max', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 1, 2], values: [200., 100., 300., 400., 600., 6.]});
  });

  it('reduceMax axes2 keep dims', function() {
    testReduce(
        'Max', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2, 1], values: [100., 200., 300., 400., 500., 600.]});
  });

  it('reduceMean default', function() {
    testReduce(
        'Mean', {}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [], values: [18.25]});
  });

  it('reduceMean default axes keep dims', function() {
    testReduce(
        'Mean', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [1, 1, 1], values: [18.25]});
  });

  it('reduceMean axes0 do not keep dims', function() {
    testReduce(
        'Mean', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [2, 2], values: [30., 1., 40., 2.]});
  });

  it('reduceMean axes1 do not keep dims', function() {
    testReduce(
        'Mean', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 2], values: [12.5, 1.5, 35., 1.5, 57.5, 1.5]});
  });

  it('reduceMean axes2 do not keep dims', function() {
    testReduce(
        'Mean', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 2], values: [3., 11., 15.5, 21., 28., 31.]});
  });

  it('reduceMean 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'Mean',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [5., 1., 20., 2.],
        },
        {
          shape: [1, 2],
          values: [3., 11.],
        });
  });

  it('reduceMean axes0 keep dims', function() {
    testReduce(
        'Mean', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [1, 2, 2], values: [30., 1., 40., 2.]});
  });

  it('reduceMean axes1 keep dims', function() {
    testReduce(
        'Mean', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 1, 2], values: [12.5, 1.5, 35., 1.5, 57.5, 1.5]});
  });

  it('reduceMean axes2 keep dims', function() {
    testReduce(
        'Mean', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 2, 1], values: [3., 11., 15.5, 21., 28., 31.]});
  });

  it('reduceMin default', function() {
    testReduce(
        'Min', {}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [], values: [1.]});
  });

  it('reduceMin default axes keep dims', function() {
    testReduce(
        'Min', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [1, 1, 1], values: [1.]});
  });

  it('reduceMin axes0 do not keep dims', function() {
    testReduce(
        'Min', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [2, 2], values: [1., 3., 4., 2.]});
  });

  it('reduceMin axes1 do not keep dims', function() {
    testReduce(
        'Min', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2], values: [1., 2., 4., 3., 500., 5.]});
  });

  it('reduceMin axes2 do not keep dims', function() {
    testReduce(
        'Min', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2], values: [1., 2., 3., 4., 5., 6.]});
  });

  it('reduceMin 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'Min',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [1., 100., 200., 2.],
        },
        {
          shape: [1, 2],
          values: [1., 2.],
        });
  });

  it('reduceMin axes0 keep dims', function() {
    testReduce(
        'Min', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [1, 2, 2], values: [1., 3., 4., 2.]});
  });

  it('reduceMin axes1 keep dims', function() {
    testReduce(
        'Min', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 1, 2], values: [1., 2., 4., 3., 500., 5.]});
  });

  it('reduceMin axes2 keep dims', function() {
    testReduce(
        'Min', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2, 1], values: [1., 2., 3., 4., 5., 6.]});
  });

  it('reduceProduct default', function() {
    testReduce(
        'Product', {}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {shape: [], values: [0.]});
  });

  it('reduceProduct default axes keep dims', function() {
    testReduce(
        'Product', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {shape: [1, 1, 1], values: [0.]});
  });

  it('reduceProduct axes0 do not keep dims', function() {
    testReduce(
        'Product', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [2, 2],
          values: [0., 45., 120., 231.],
        });
  });

  it('reduceProduct axes1 do not keep dims', function() {
    testReduce(
        'Product', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2],
          values: [0., 3., 24., 35., 80., 99.],
        });
  });

  it('reduceProduct axes2 do not keep dims', function() {
    testReduce(
        'Product', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2],
          values: [0., 6., 20., 42., 72., 110.],
        });
  });

  it('reduceProduct 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'Product',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., 1., 2., 3.],
        },
        {
          shape: [1, 2],
          values: [0., 6.],
        });
  });

  it('reduceProduct axes0 keep dims', function() {
    testReduce(
        'Product', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [1, 2, 2],
          values: [0., 45., 120., 231.],
        });
  });

  it('reduceProduct axes1 keep dims', function() {
    testReduce(
        'Product', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 1, 2],
          values: [0., 3., 24., 35., 80., 99.],
        });
  });

  it('reduceProduct axes2 keep dims', function() {
    testReduce(
        'Product', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2, 1],
          values: [0., 6., 20., 42., 72., 110.],
        });
  });

  it('reduceSum default', function() {
    testReduce(
        'Sum', {}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {shape: [], values: [66.]});
  });

  it('reduceSum default axes keep dims', function() {
    testReduce(
        'Sum', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {shape: [1, 1, 1], values: [66.]});
  });

  it('reduceSum axes0 do not keep dims', function() {
    testReduce(
        'Sum', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [2, 2],
          values: [12., 15., 18., 21.],
        });
  });

  it('reduceSum axes1 do not keep dims', function() {
    testReduce(
        'Sum', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2],
          values: [2., 4., 10., 12., 18., 20.],
        });
  });

  it('reduceSum axes2 do not keep dims', function() {
    testReduce(
        'Sum', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2],
          values: [1., 5., 9., 13., 17., 21.],
        });
  });

  it('reduceSum 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'Sum',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., 1., 2., 3.],
        },
        {
          shape: [1, 2],
          values: [1., 5.],
        });
  });

  it('reduceSum axes0 keep dims', function() {
    testReduce(
        'Sum', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [1, 2, 2],
          values: [12., 15., 18., 21.],
        });
  });

  it('reduceSum axes1 keep dims', function() {
    testReduce(
        'Sum', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 1, 2],
          values: [2., 4., 10., 12., 18., 20.],
        });
  });

  it('reduceSum axes2 keep dims', function() {
    testReduce(
        'Sum', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2, 1],
          values: [1., 5., 9., 13., 17., 21.],
        });
  });

  it('reduceSumSquare default', function() {
    testReduce(
        'SumSquare', {}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [], values: [506]});
  });

  it('reduceSumSquare default axes keep dims', function() {
    testReduce(
        'SumSquare', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [1, 1, 1], values: [506]});
  });

  it('reduceSumSquare axes0 do not keep dims', function() {
    testReduce(
        'SumSquare', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [2, 2],
          values: [80, 107, 140, 179],
        });
  });

  it('reduceSumSquare axes1 do not keep dims', function() {
    testReduce(
        'SumSquare', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [4, 10, 52, 74, 164, 202],
        });
  });

  it('reduceSumSquare axes2 do not keep dims', function() {
    testReduce(
        'SumSquare', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [1, 13, 41, 85, 145, 221],
        });
  });

  it('reduceSumSquare 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'SumSquare',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., 1., 2., 3.],
        },
        {
          shape: [1, 2],
          values: [1, 13],
        });
  });

  it('reduceSumSquare axes0 keep dims', function() {
    testReduce(
        'SumSquare', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [80, 107, 140, 179],
        });
  });

  it('reduceSumSquare axes1 keep dims', function() {
    testReduce(
        'SumSquare', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [4, 10, 52, 74, 164, 202],
        });
  });

  it('reduceSumSquare axes2 keep dims', function() {
    testReduce(
        'SumSquare', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [1, 13, 41, 85, 145, 221],
        });
  });

  it('reduceL1 default', function() {
    testReduce(
        'L1', {}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {shape: [], values: [66.]});
  });

  it('reduceL1 default axes keep dims', function() {
    testReduce(
        'L1', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {shape: [1, 1, 1], values: [66.]});
  });

  it('reduceL1 axes0 do not keep dims', function() {
    testReduce(
        'L1', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {
          shape: [2, 2],
          values: [12., 15., 18., 21.],
        });
  });

  it('reduceL1 axes1 do not keep dims', function() {
    testReduce(
        'L1', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {
          shape: [3, 2],
          values: [2., 4., 10., 12., 18., 20.],
        });
  });

  it('reduceL1 axes2 do not keep dims', function() {
    testReduce(
        'L1', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {
          shape: [3, 2],
          values: [1., 5., 9., 13., 17., 21.],
        });
  });

  it('reduceL1 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'L1',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., -1., 2., -3.],
        },
        {
          shape: [1, 2],
          values: [1., 5.],
        });
  });

  it('reduceL1 axes0 keep dims', function() {
    testReduce(
        'L1', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [12., 15., 18., 21.],
        });
  });

  it('reduceL1 axes1 keep dims', function() {
    testReduce(
        'L1', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [2., 4., 10., 12., 18., 20.],
        });
  });

  it('reduceL1 axes2 keep dims', function() {
    testReduce(
        'L1', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., -1., 2., -3.,
            4., -5., 6., -7.,
            8., -9., 10., -11.,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [1., 5., 9., 13., 17., 21.],
        });
  });

  it('reduceL2 default', function() {
    testReduce(
        'L2', {}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [], values: [22.494443758403985]});
  });

  it('reduceL2 default axes keep dims', function() {
    testReduce(
        'L2', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [1, 1, 1], values: [22.494443758403985]});
  });

  it('reduceL2 axes0 do not keep dims', function() {
    testReduce(
        'L2', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [2, 2],
          values: [
            8.94427190999916,
            10.344080432788601,
            11.832159566199232,
            13.379088160259652,
          ],
        });
  });

  it('reduceL2 axes1 do not keep dims', function() {
    testReduce(
        'L2', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [
            2,
            3.1622776601683795,
            7.211102550927978,
            8.602325267042627,
            12.806248474865697,
            14.212670403551895,
          ],
        });
  });

  it('reduceL2 axes2 do not keep dims', function() {
    testReduce(
        'L2', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [
            1,
            3.605551275463989,
            6.4031242374328485,
            9.219544457292887,
            12.041594578792296,
            14.866068747318506,
          ],
        });
  });

  it('reduceL2 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'L2',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., 1., 2., 3.],
        },
        {
          shape: [1, 2],
          values: [1, 3.605551275463989],
        });
  });

  it('reduceL2 axes0 keep dims', function() {
    testReduce(
        'L2', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            8.94427190999916,
            10.344080432788601,
            11.832159566199232,
            13.379088160259652,
          ],
        });
  });

  it('reduceL2 axes1 keep dims', function() {
    testReduce(
        'L2', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [
            2,
            3.1622776601683795,
            7.211102550927978,
            8.602325267042627,
            12.806248474865697,
            14.212670403551895,
          ],
        });
  });

  it('reduceL2 axes2 keep dims', function() {
    testReduce(
        'L2', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [
            1,
            3.605551275463989,
            6.4031242374328485,
            9.219544457292887,
            12.041594578792296,
            14.866068747318506,
          ],
        });
  });

  it('reduceLogSum default', function() {
    testReduce(
        'LogSum', {}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [], values: [4.189654742026425]});
  });

  it('reduceLogSum default axes keep dims', function() {
    testReduce(
        'LogSum', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [1, 1, 1], values: [4.189654742026425]});
  });

  it('reduceLogSum axes0 do not keep dims', function() {
    testReduce(
        'LogSum', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [2, 2],
          values: [
            2.4849066497880004,
            2.70805020110221,
            2.8903717578961645,
            3.044522437723423,
          ],
        });
  });

  it('reduceLogSum axes1 do not keep dims', function() {
    testReduce(
        'LogSum', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [
            0.6931471805599453,
            1.3862943611198906,
            2.302585092994046,
            2.4849066497880004,
            2.8903717578961645,
            2.995732273553991,
          ],
        });
  });

  it('reduceLogSum axes2 do not keep dims', function() {
    testReduce(
        'LogSum', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [
            0,
            1.6094379124341003,
            2.1972245773362196,
            2.5649493574615367,
            2.833213344056216,
            3.044522437723423,
          ],
        });
  });

  it('reduceLogSum 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'LogSum',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., 1., 2., 3.],
        },
        {
          shape: [1, 2],
          values: [0, 1.6094379124341003],
        });
  });

  it('reduceLogSum axes0 keep dims', function() {
    testReduce(
        'LogSum', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            2.4849066497880004,
            2.70805020110221,
            2.8903717578961645,
            3.044522437723423,
          ],
        });
  });

  it('reduceLogSum axes1 keep dims', function() {
    testReduce(
        'LogSum', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [
            0.6931471805599453,
            1.3862943611198906,
            2.302585092994046,
            2.4849066497880004,
            2.8903717578961645,
            2.995732273553991,
          ],
        });
  });

  it('reduceLogSum axes2 keep dims', function() {
    testReduce(
        'LogSum', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [
            0,
            1.6094379124341003,
            2.1972245773362196,
            2.5649493574615367,
            2.833213344056216,
            3.044522437723423,
          ],
        });
  });

  it('reduceLogSumExp default', function() {
    testReduce(
        'LogSumExp', {}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [], values: [11.458669001155853]});
  });

  it('reduceLogSumExp default axes keep dims', function() {
    testReduce(
        'LogSumExp', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {shape: [1, 1, 1], values: [11.458669001155853]});
  });

  it('reduceLogSumExp axes0 do not keep dims', function() {
    testReduce(
        'LogSumExp', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [2, 2],
          values: [
            8.018479302594658,
            9.018479302594658,
            10.018479302594658,
            11.018479302594658,
          ],
        });
  });

  it('reduceLogSumExp axes1 do not keep dims', function() {
    testReduce(
        'LogSumExp', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [
            2.1269280110429727,
            3.1269280110429722,
            6.126928011042972,
            7.126928011042972,
            10.126928011042972,
            11.126928011042972,
          ],
        });
  });

  it('reduceLogSumExp axes2 do not keep dims', function() {
    testReduce(
        'LogSumExp', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2],
          values: [
            1.3132616875182228,
            3.313261687518223,
            5.313261687518223,
            7.313261687518223,
            9.313261687518223,
            11.313261687518223,
          ],
        });
  });

  it('reduceLogSumExp 3D of shape [1, 2, 2] axes2 do not keep dims', function() {
    testReduce(
        'LogSumExp',
        {
          axes: [2],
          keepDimensions: false,
        },
        {
          shape: [1, 2, 2],
          values: [0., 1., 2., 3.],
        },
        {
          shape: [1, 2],
          values: [1.3132616875182228, 3.313261687518223],
        });
  });

  it('reduceLogSumExp axes0 keep dims', function() {
    testReduce(
        'LogSumExp', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            8.018479302594658,
            9.018479302594658,
            10.018479302594658,
            11.018479302594658,
          ],
        });
  });

  it('reduceLogSumExp axes1 keep dims', function() {
    testReduce(
        'LogSumExp', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [
            2.1269280110429727,
            3.1269280110429722,
            6.126928011042972,
            7.126928011042972,
            10.126928011042972,
            11.126928011042972,
          ],
        });
  });

  it('reduceLogSumExp axes2 keep dims', function() {
    testReduce(
        'LogSumExp', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0., 1., 2., 3.,
            4., 5., 6., 7.,
            8., 9., 10., 11.,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [
            1.3132616875182228,
            3.313261687518223,
            5.313261687518223,
            7.313261687518223,
            9.313261687518223,
            11.313261687518223,
          ],
        });
  });
});
