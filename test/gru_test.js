'use strict';

import {gru} from '../src/gru.js';
import {Tensor} from '../src/lib/tensor.js';
import {relu} from '../src/relu.js';
import * as utils from './utils.js';

describe('test gru', function() {
  it('gru with 1 step', function() {
    const steps = 1;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = new Tensor([steps, batchSize, inputSize], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        [
          0.3148022,
          -0.4366297,
          -0.9718124,
          1.9853785,
          2.2497437,
          0.6179927,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]);
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(2));
    const resetAfter = true;
    const layout = 'rzn';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, resetAfter, layout});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      1.9801673183552388,
      1.9812534682811542,
      1.9376592706336329,
      1.9935192730591977,
      1.9947569570033654,
      1.9759958501762682,
      1.997469445392646,
      1.9980404252433588,
      1.9902071255213296,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with 2 steps', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with explict returnSequence false', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const returnSequence = false;
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, returnSequence});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with returnSequence true', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const returnSequence = true;
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, returnSequence});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    utils.checkShape(outputs[1], [steps, numDirections, batchSize, hiddenSize]);
    const expected = [
      [
        0.22391088955449673,
        0.22391088955449673,
        0.22391088955449673,
        0.22391088955449673,
        0.22391088955449673,
        0.16530139937319663,
        0.16530139937319663,
        0.16530139937319663,
        0.16530139937319663,
        0.16530139937319663,
        0.0797327116380732,
        0.0797327116380732,
        0.0797327116380732,
        0.0797327116380732,
        0.0797327116380732,
      ],
      [
        0.20053661855501925,
        0.20053661855501925,
        0.20053661855501925,
        0.20053661855501925,
        0.20053661855501925,
        0.15482337214048048,
        0.15482337214048048,
        0.15482337214048048,
        0.15482337214048048,
        0.15482337214048048,
        0.07484276504070396,
        0.07484276504070396,
        0.07484276504070396,
        0.07484276504070396,
        0.07484276504070396,
        0.22391088955449673,
        0.22391088955449673,
        0.22391088955449673,
        0.22391088955449673,
        0.22391088955449673,
        0.16530139937319663,
        0.16530139937319663,
        0.16530139937319663,
        0.16530139937319663,
        0.16530139937319663,
        0.0797327116380732,
        0.0797327116380732,
        0.0797327116380732,
        0.0797327116380732,
        0.0797327116380732,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(outputs[i], expected[i]);
    }
  });

  it('gru with explict forward direction', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'forward';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with backward direction', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'backward';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22227008136062426,
      0.22227008136062426,
      0.22227008136062426,
      0.22227008136062426,
      0.22227008136062426,
      0.1652493513699554,
      0.1652493513699554,
      0.1652493513699554,
      0.1652493513699554,
      0.1652493513699554,
      0.07972921857068853,
      0.07972921857068853,
      0.07972921857068853,
      0.07972921857068853,
      0.07972921857068853,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with both direction', function() {
    const steps = 2;
    const numDirections = 2;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const initialHiddenState = new Tensor([numDirections, batchSize, hiddenSize],
        new Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'both';
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.22227008136062426,
      0.22227008136062426,
      0.22227008136062426,
      0.22227008136062426,
      0.22227008136062426,
      0.1652493513699554,
      0.1652493513699554,
      0.1652493513699554,
      0.1652493513699554,
      0.1652493513699554,
      0.07972921857068853,
      0.07972921857068853,
      0.07972921857068853,
      0.07972921857068853,
      0.07972921857068853,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru without initialHiddenState', function() {
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        new Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        new Array(numDirections * 3 * hiddenSize).fill(0));
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.22391088955449673,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.16530139937319663,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
      0.0797327116380732,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with resetAfter=true', function() {
    const steps = 1;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 2;
    const hiddenSize = 4;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 2, 1, 1, 1]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        [
          1,   -1,   2, -2,  0.5, -0.5, 0, 0.1, 1,   -1,   2, -2,
          0.5, -0.5, 0, 0.1, 1,   -1,   2, -2,  0.5, -0.5, 0, 0.1,
        ]);
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        [1, 2, 1, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]);
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        [1, 2, 1, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]);
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, resetAfter: true, activations: [relu, relu]});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected = [
      0, 0, -0.375, -5.760000000000001,
      -6, -22.5, -3.375, -5.115,
      -1.5, -4.5, -1.5, -5.115,
    ];
    utils.checkValue(outputs[0], expected);
  });

  it('gru with 2 steps, both direction and resetAfter=true', function() {
    const steps = 2;
    const numDirections = 2;
    const batchSize = 3;
    const inputSize = 2;
    const hiddenSize = 4;
    const input = new Tensor([steps, batchSize, inputSize],
        [1, 2, 2, 1, 1, 1, 3, 4, 1, 2, 1, 1]);
    const weight = new Tensor([numDirections, 3 * hiddenSize, inputSize],
        [
          1, -1, 2, -2, 0.5, -0.5, 0, 0.1, 1, -1, 2, -2, 0.5, -0.5, 0, 0.1,
          1, -1, 2, -2, 0.5, -0.5, 0, 0.1, 1, -1, 2, -2, 0.5, -0.5, 0, 0.1,
          1, -1, 2, -2, 0.5, -0.5, 0, 0.1, 1, -1, 2, -2, 0.5, -0.5, 0, 0.1,
        ]);
    const recurrentWeight = new Tensor([numDirections, 3 * hiddenSize, hiddenSize],
        new Array(numDirections * 3 * hiddenSize * hiddenSize).fill(0.1));
    const bias = new Tensor([numDirections, 3 * hiddenSize],
        [
          1, 2, 1, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5,
          1, 2, 1, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5,
        ]);
    const recurrentBias = new Tensor([numDirections, 3 * hiddenSize],
        [
          1, 2, 1, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5,
          1, 2, 1, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5,
        ]);
    const outputs = gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, resetAfter: true, returnSequence: true, direction: 'both',
          activations: [relu, relu]});
    utils.checkShape(outputs[0], [numDirections, batchSize, hiddenSize]);
    const expected0 = [
      0,
      0,
      -0.33243749999999994,
      -23.753077664625003,
      0,
      0,
      0,
      -2.2133150000000024,
      -1.1077499999999996,
      -12.323249999999998,
      -1.1077499999999996,
      -14.5189275,
      0,
      0,
      -0.2806875,
      -25.444256959125006,
      -1.7041918646249996,
      -9.280220864624999,
      -1.4041221146249998,
      -21.154623314625002,
      -1.1077499999999996,
      -12.323249999999998,
      -1.1077499999999996,
      -14.5189275,
    ];
    utils.checkValue(outputs[0], expected0);

    utils.checkShape(outputs[1], [steps, numDirections, batchSize, hiddenSize]);
    const expected1 = [
      0,
      0,
      -0.375,
      -5.760000000000001,
      -6,
      -22.5,
      -3.375,
      -5.115,
      -1.5,
      -4.5,
      -1.5,
      -5.115,
      0,
      0,
      -0.2806875,
      -25.444256959125006,
      -1.7041918646249996,
      -9.280220864624999,
      -1.4041221146249998,
      -21.154623314625002,
      -1.1077499999999996,
      -12.323249999999998,
      -1.1077499999999996,
      -14.5189275,
      0,
      0,
      -0.33243749999999994,
      -23.753077664625003,
      0,
      0,
      0,
      -2.2133150000000024,
      -1.1077499999999996,
      -12.323249999999998,
      -1.1077499999999996,
      -14.5189275,
      0,
      0,
      -0.375,
      -7.1400000000000015,
      0,
      0,
      -0.375,
      -5.760000000000001,
      -1.5,
      -4.5,
      -1.5,
      -5.115,
    ];
    utils.checkValue(outputs[1], expected1);
  });
});
