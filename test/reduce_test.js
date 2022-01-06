'use strict';

import * as reducers from '../src/reduce.js';
import {Tensor} from '../src/tensor.js';

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

  it('reduceMax negative axes do not keep dims', function() {
    testReduce(
        'Max', {axes: [-1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2], values: [100., 200., 300., 400., 500., 600.]});
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

  it('reduceMax negative axes keep dims', function() {
    testReduce(
        'Max', {axes: [-1], keepDimensions: true}, {
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

  it('reduceMean negative axes do not keep dims', function() {
    testReduce(
        'Mean', {axes: [-1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 2], values: [3., 11., 15.5, 21., 28., 31.]});
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

  it('reduceMean negative axes keep dims', function() {
    testReduce(
        'Mean', {axes: [-1], keepDimensions: true}, {
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

  it('reduceMin negative axes do not keep dims', function() {
    testReduce(
        'Min', {axes: [-1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1., 100., 200., 2., 300., 3., 4., 400., 500., 5., 600., 6.],
        },
        {shape: [3, 2], values: [1., 2., 3., 4., 5., 6.]});
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

  it('reduceMin negative axes keep dims', function() {
    testReduce(
        'Min', {axes: [-1], keepDimensions: true}, {
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

  it('reduceProduct negative axes do not keep dims', function() {
    testReduce(
        'Product', {axes: [-1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2],
          values: [0., 6., 20., 42., 72., 110.],
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

  it('reduceProduct negative axes keep dims', function() {
    testReduce(
        'Product', {axes: [-1], keepDimensions: true}, {
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

  it('reduceSum negative axes do not keep dims', function() {
    testReduce(
        'Sum', {axes: [-1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2],
          values: [1., 5., 9., 13., 17., 21.],
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

  it('reduceSum negative axes keep dims', function() {
    testReduce(
        'Sum', {axes: [-1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        },
        {
          shape: [3, 2, 1],
          values: [1., 5., 9., 13., 17., 21.],
        });
  });
});
