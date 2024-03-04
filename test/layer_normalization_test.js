'use strict';

import {layerNormalization} from '../src/layer_normalization.js';
import {Tensor} from '../src/lib/tensor.js';
import * as utils from './utils.js';


describe('test layerNormalization', function() {
  function testLayerNorm(
      input, expected, scale = undefined, bias = undefined, axes = undefined, options = {}) {
    const inputTensor = new Tensor(input.shape, input.value);
    if (scale) {
      options.scale = new Tensor(scale.shape, scale.value);
    }
    if (bias) {
      options.bias = new Tensor(bias.shape, bias.value);
    }
    if (axes) {
      options.axes = axes;
    }
    const outputTensor = layerNormalization(inputTensor, options);
    utils.checkShape(outputTensor, input.shape);
    utils.checkValue(outputTensor, expected);
  }

  it('layerNormalization default 2D', function() {
    testLayerNorm(
        {
          shape: [2, 3],
          value: [1, 2, 3, 3, 6, 24],
        },
        [
          -1.2247356859083902,
          0,
          1.2247356859083902,
          -0.8626621354727335,
          -0.5391638346704585,
          1.4018259701431919,
        ],
    );
  });

  it('layerNormalization default 3D', function() {
    testLayerNorm(
        {
          shape: [2, 2, 3],
          value: [1, 2, 3, 6, 5, 4, 3, 6, 24, -10, 0, 5],
        },
        [
          -1.4638475999719223,
          -0.8783085599831534,
          -0.29276951999438444,
          1.4638475999719223,
          0.8783085599831534,
          0.29276951999438444,
          -0.1645769966453613,
          0.131661597316289,
          1.9090931610861905,
          -1.4482775704791793,
          -0.46081559060701155,
          0.032915399329072226,
        ],
    );
  });

  it('layerNormalization default 3D with axes=[2]', function() {
    testLayerNorm(
        {
          shape: [2, 2, 3],
          value: [1, 2, 3, 6, 5, 4, 3, 6, 24, -10, 0, 5],
        },
        [
          -1.2247356859083902,
          0,
          1.2247356859083902,
          1.2247356859083902,
          0,
          -1.2247356859083902,
          -0.8626621354727335,
          -0.5391638346704585,
          1.4018259701431919,
          -1.3363060377513567,
          0.26726120755027133,
          1.0690448302010853,
        ],
        undefined,
        undefined,
        [2],
    );
  });

  it('layerNormalization 3D with scale and axes=[2]', function() {
    testLayerNorm(
        {
          shape: [2, 2, 3],
          value: [1, 2, 3, 6, 5, 4, 3, 6, 24, -10, 0, 5],
        },
        [
          -2.4494713718167804,
          0, 4.898942743633561,
          2.4494713718167804,
          0, -4.898942743633561,
          -1.725324270945467,
          -1.6174915040113755,
          5.6073038805727675,
          -2.6726120755027134,
          0.8017836226508139,
          4.276179320804341,
        ],
        {
          shape: [3],
          value: [2, 3, 4],
        },
        undefined,
        [2],
    );
  });

  it('layerNormalization 3D with scale and bias and axes=[2]', function() {
    testLayerNorm(
        {
          shape: [2, 2, 3],
          value: [1, 2, 3, 6, 5, 4, 3, 6, 24, -10, 0, 5],
        },
        [
          -1.4494713718167804,
          2,
          7.898942743633561,
          3.4494713718167804,
          2,
          -1.8989427436335609,
          -0.725324270945467,
          0.38250849598862446,
          8.607303880572768,
          -1.6726120755027134,
          2.801783622650814,
          7.276179320804341,
        ],
        {
          shape: [3],
          value: [2, 3, 4],
        },
        {
          shape: [3],
          value: [1, 2, 3],
        },
        [2],
    );
  });

  it('layerNormalization 3D with epsilon', function() {
    testLayerNorm(
        {
          shape: [2, 2, 3],
          value: [1, 2, 3, 6, 5, 4, 3, 6, 24, -10, 0, 5],
        },
        [
          -1.4635992282002273,
          -0.8781595369201364,
          -0.29271984564004544,
          1.4635992282002273,
          0.8781595369201364,
          0.29271984564004544,
          -0.16457620229526346,
          0.1316609618362107,
          1.9090839466250555,
          -1.4482705801983182,
          -0.46081336642673765,
          0.032915240459052655,
        ],
        undefined,
        undefined,
        undefined,
        {
          epsilon: 1e-3,
        },
    );
  });

  it('layerNormalization test descending order axes', function() {
    testLayerNorm(
        {
          shape: [1, 2, 3],
          value: [1, 2, 3, 4, 5, 6],
        },
        [
          -1.4638475999719223,
          -2.6349256799494603,
          -1.4638475999719223,
          0.5855390399887689,
          3.5132342399326135,
          8.783085599831534,
        ],
        {
          shape: [3, 2],
          value: [1, 2, 3, 4, 5, 6],
        },
        undefined,
        [2, 1],
    );
  });

  it('layerNormalization Ascending order axis', function() {
    testLayerNorm(
        {
          shape: [1, 2, 3],
          value: [1, 2, 3, 4, 5, 6],
        },
        [
          -1.4638475999719223,
          -1.7566171199663068,
          -0.8783085599831533,
          1.1710780799775378,
          4.391542799915767,
          8.783085599831534,
        ],
        {
          shape: [2, 3],
          value: [1, 2, 3, 4, 5, 6],
        },
        undefined,
        [1, 2],
    );
  });
});
