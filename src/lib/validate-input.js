'use strict';

import {sizeOfShape} from './tensor.js';

/**
 * Check the tensor whether it is a 1-D tensor and its length is equal to `expectedSize`.
 * @param {Tensor} a
 * @param {Number} expectedSize
 * @param {String} name
 */
function check1DTensorWithSize(a, expectedSize, name) {
  if (a) {
    if (a.rank !== 1) {
      throw new Error(`The parameter ${name} is not a 1-D tensor.`);
    } else {
      if (a.shape[0] !== expectedSize) {
        throw new Error(`The length ${a.shape[0]} of the ${name} values is not equal to the ` +
          `size ${expectedSize} of the input dimension denoted by options.axis.`);
      }
    }
  }
}

export function validateBatchNormalizationParams(input, mean, variance,
    {axis=1, scale, bias} = {}) {
  if (!Number.isInteger(axis)) {
    throw new Error(`Invalid axis ${axis} - axis should be an integer.`);
  }
  const dim = input.shape[axis];
  check1DTensorWithSize(mean, dim, 'mean');
  check1DTensorWithSize(variance, dim, 'variance');
  check1DTensorWithSize(scale, dim, 'scale');
  check1DTensorWithSize(bias, dim, 'bias');
}

export function validateLayerNormalizationParams(input, {axes, scale, bias} = {}) {
  if (scale && axes) {
    if (scale.rank !== axes.length) {
      throw new Error('DataError: the rank of scale is not equal to the size of axes.');
    }
  }
  if (bias && axes) {
    if (bias.rank !== axes.length) {
      throw new Error('DataError: the rank of bias is not equal to the size of axes.');
    }
  }
  if (axes) {
    for (let i = 0; i < axes.length; i++) {
      const axis = axes[i];
      if (axis >= input.rank) {
        throw new Error('DataError: the value of axis in axes should be smaller than input.rank');
      }
      const dim = input.shape[axis];
      if (scale) {
        if (scale.shape[i] !== dim) {
          throw new Error(`The length ${scale.shape[i]} of the scale values is not equal to the ` +
          `size ${dim} of the input dimension denoted by options.axis.`);
        }
      }
      if (bias) {
        if (bias.shape[i] !== dim) {
          throw new Error(`The length ${bias.shape[i]} of the bias values is not equal to the ` +
          `size ${dim} of the input dimension denoted by options.axis.`);
        }
      }
    }
  }
}

export function validateInstanceNormalizationParams(
    input,
    {
      scale,
      bias,
      epsilon=1e-5,
      layout='nchw',
    } = {}) {
  if (input.rank !== 4) {
    throw new Error('The input should be a 4-D tensor.');
  }
  if (layout !== 'nchw' && layout !== 'nhwc') {
    throw new Error(`The layout ${layout} is invalid.`);
  }
  const axis = layout === 'nchw' ? 1 : 3;
  const dim = input.shape[axis];
  check1DTensorWithSize(scale, dim, 'scale');
  check1DTensorWithSize(bias, dim, 'bias');
  if (typeof epsilon !== 'number') {
    throw new Error(`The epsilon ${epsilon} should be a number.`);
  }
}

export function validateConcatParams(inputs, axis) {
  const rank = inputs[0].rank;
  if (!Number.isInteger(axis)) {
    throw new Error(`Invalid axis ${axis} - axis should be an integer.`);
  } else {
    if (axis < 0 || axis >= rank) {
      throw new Error(`Invalid axis ${axis} - axis should be in the interval [0, ${rank}).`);
    }
  }
  const inputShape = inputs[0].shape;
  for (let i = 1; i < inputs.length; ++i) {
    if (inputs[i].rank !== rank) {
      throw new Error('All input tensors should have the same rank.');
    } else {
      const shape = inputs[i].shape;
      for (let j = 0; j < inputShape.length; ++j) {
        if (j !== axis) {
          if (inputShape[j] !== shape[j]) {
            throw new Error('All input tensors should have the same shape, ' +
             'except for the size of the dimension to concatenate on.');
          }
        }
      }
    }
  }
}

export function validateConv2dParams(input, filter, {bias, groups = 1}) {
  const inputChannels = input.shape[1];
  const outputChannels = filter.shape[0];
  const filterInputChannels = filter.shape[1];
  if (input.rank !== 4) {
    throw new Error('The input should be a 4-D tensor.');
  }
  if (filter.rank !== 4) {
    throw new Error('The filter should be a 4-D tensor.');
  }
  if (inputChannels !== filterInputChannels * groups) {
    throw new Error('The input channels of filter is invalid.');
  }
  if (bias && (bias.rank !== 1 || bias.shape[0] != outputChannels)) {
    throw new Error('the bias should be a 1-D tensor with the shape of [output_channels].');
  }
}

export function validateConvTranspose2dParams(input, filter, {bias, groups = 1}) {
  const inputChannels = input.shape[1];
  // filter of oihw
  const outputChannelsPerGroup = filter.shape[0];
  const outputChannels = outputChannelsPerGroup * groups;

  if (input.rank !== 4) {
    throw new Error('The input should be a 4-D tensor.');
  }
  if (filter.rank !== 4) {
    throw new Error('The filter should be a 4-D tensor.');
  }
  if (inputChannels % groups !== 0) {
    throw new Error('The input channels is invalid.');
  }
  if (inputChannels !== filter.shape[1]) {
    throw new Error('The input channels of filter is invalid.');
  }
  if (bias && (bias.rank !== 1 || bias.shape[0] != outputChannels)) {
    throw new Error('the bias should be a 1-D tensor with the shape of [output_channels].');
  }
}

export function validateGemmParams(a, b) {
  if (a.rank !== 2) {
    throw new Error('The input a is not a 2-D tensor.');
  }
  if (b.rank !== 2) {
    throw new Error('The input b is not a 2-D tensor.');
  }
}

export function validateLstmCellParams(input, weight, recurrentWeight,
    hiddenState, cellState, hiddenSize,
    {bias, recurrentBias, peepholeWeight, layout = 'iofg'} = {}) {
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
  }
  if (input.rank !== 2) {
    throw new Error(`The input (rank ${input.rank}) is not a 2-D tensor.`);
  }
  const batchSize = input.shape[0];
  const inputSize = input.shape[1];
  if (weight.rank !== 2) {
    throw new Error(`The weight (rank ${weight.rank}) is not a 2-D tensor.`);
  }
  if (weight.shape[0] !== 4 * hiddenSize || weight.shape[1] !== inputSize) {
    throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]}] is invalid.`);
  }
  if (recurrentWeight.rank !== 2) {
    throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 2-D tensor.`);
  }
  if (recurrentWeight.shape[0] !== 4 * hiddenSize || recurrentWeight.shape[1] !== hiddenSize) {
    throw new Error(`The shape of recurrentWeight ` +
    `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}] is invalid.`);
  }
  if (hiddenState.rank !== 2) {
    throw new Error(`The hiddenState (rank ${hiddenState.rank}) is not a 2-D tensor.`);
  }
  if (hiddenState.shape[0] !== batchSize || hiddenState.shape[1] !== hiddenSize) {
    throw new Error(`The shape of hiddenState` +
                    `[${hiddenState.shape[0]}, ${hiddenState.shape[1]}] is invalid.`);
  }
  if (cellState.rank !== 2) {
    throw new Error(`The cellState (rank ${cellState.rank}) is not a 2-D tensor.`);
  }
  if (cellState.shape[0] !== batchSize || cellState.shape[1] !== hiddenSize) {
    throw new Error(`The shape of cellState [${cellState.shape[0]},
     ${cellState.shape[1]}] is invalid.`);
  }
  if (bias) {
    if (bias.rank !== 1) {
      throw new Error(`The bias (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (bias.shape[0] !== 4 * hiddenSize) {
      throw new Error(`The shape of bias [${bias.shape[0]}] is invalid.`);
    }
  }
  if (recurrentBias) {
    if (recurrentBias.rank !== 1) {
      throw new Error(`The recurrentBias (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (recurrentBias.shape[0] !== 4 * hiddenSize) {
      throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]}] is invalid.`);
    }
  }
  if (peepholeWeight) {
    if (peepholeWeight.rank !== 1) {
      throw new Error(`The peepholeWeight (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (peepholeWeight.shape[0] !== 3 * hiddenSize) {
      throw new Error(`The shape of peepholeWeight [${peepholeWeight.shape[0]}] is invalid.`);
    }
  }
  if (layout !== 'iofg' && layout !== 'ifgo') {
    throw new Error(`The layout ${layout} is invalid.`);
  }
}

export function validateLstmParams(input, weight, recurrentWeight, steps, hiddenSize,
    {bias, recurrentBias, peepholeWeight, initialHiddenState, initialCellState,
      direction = 'forward', layout = 'iofg'}) {
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new Error(`The steps ${steps} is invalid.`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
  }
  if (input.rank !== 3) {
    throw new Error(`The input (rank ${input.rank}) is not a 3-D tensor.`);
  }
  if (input.shape[0] !== steps) {
    throw new Error(`The input.shape[0] ${input.shape[0]} is not equal to steps ${steps}.`);
  }
  const batchSize = input.shape[1];
  const inputSize = input.shape[2];
  if (direction !== 'forward' && direction !== 'backward' && direction !== 'both') {
    throw new Error(`The direction ${direction} is invalid.`);
  }
  const numDirections = (direction === 'both' ? 2 : 1);
  if (weight.rank !== 3) {
    throw new Error(`The weight (rank ${weight.rank}) is not a 3-D tensor.`);
  }
  if (weight.shape[0] !== numDirections || weight.shape[1] !== 4 * hiddenSize ||
    weight.shape[2] !== inputSize) {
    throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]},
      ${weight.shape[2]}] is invalid.`);
  }
  if (recurrentWeight.rank !== 3) {
    throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 3-D tensor.`);
  }
  if (recurrentWeight.shape[0] !== numDirections ||
    recurrentWeight.shape[1] !== 4 * hiddenSize ||
    recurrentWeight.shape[2] !== hiddenSize) {
    throw new Error(`The shape of recurrentWeight ` +
                  `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}, ` +
                  `${recurrentWeight.shape[2]}] is invalid.`);
  }
  if (bias) {
    if (bias.rank !== 2) {
      throw new Error(`The bias (rank ${bias.rank}) is not a 2-D tensor.`);
    }
    if (bias.shape[0] !== numDirections || bias.shape[1] !== 4 * hiddenSize) {
      throw new Error(`The shape of bias [${bias.shape[0]}, ${bias.shape[1]}] is invalid.`);
    }
  }
  if (recurrentBias) {
    if (recurrentBias.rank !== 2) {
      throw new Error(`The recurrentBias (rank ${recurrentBias.rank}) is not a 2-D tensor.`);
    }
    if (recurrentBias.shape[0] !== numDirections || recurrentBias.shape[1] !== 4 * hiddenSize) {
      throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]},
        ${recurrentBias.shape[1]}] is invalid.`);
    }
  }
  if (peepholeWeight) {
    if (peepholeWeight.rank !== 2) {
      throw new Error(`The peepholeWeight (rank ${peepholeWeight.rank}) is not a 2-D tensor.`);
    }
    if (peepholeWeight.shape[0] !== numDirections || peepholeWeight.shape[1] !== 3 * hiddenSize) {
      throw new Error(`The shape of peepholeWeight [${peepholeWeight.shape[0]},
        ${peepholeWeight.shape[1]}] is invalid.`);
    }
  }
  if (initialHiddenState) {
    if (initialHiddenState.rank !== 3) {
      throw new Error(
          `The initialHiddenState (rank ${initialHiddenState.rank}) is not a 3-D tensor.`);
    }
    if (initialHiddenState.shape[0] !== numDirections ||
      initialHiddenState.shape[1] !== batchSize ||
      initialHiddenState.shape[2] !== hiddenSize) {
      throw new Error(`The shape of initialHiddenState [${initialHiddenState.shape[0]},
        ${initialHiddenState.shape[1]}, ${initialHiddenState.shape[2]}] is invalid.`);
    }
  }
  if (initialCellState) {
    if (initialCellState.rank !== 3) {
      throw new Error(
          `The initialCellState (rank ${initialCellState.rank}) is not a 3-D tensor.`);
    }
    if (initialCellState.shape[0] !== numDirections ||
      initialCellState.shape[1] !== batchSize ||
      initialCellState.shape[2] !== hiddenSize) {
      throw new Error(`The shape of initialCellState [${initialCellState.shape[0]},
        ${initialCellState.shape[1]}, ${initialCellState.shape[2]}] is invalid.`);
    }
  }
  if (layout !== 'iofg' && layout !== 'ifgo') {
    throw new Error(`The layout ${layout} is invalid.`);
  }
}


export function validateGruCellParams(input, weight, recurrentWeight, hiddenState, hiddenSize,
    {bias, recurrentBias, layout = 'zrn'} = {}) {
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
  }
  if (input.rank !== 2) {
    throw new Error(`The input (rank ${input.rank}) is not a 2-D tensor.`);
  }
  const batchSize = input.shape[0];
  const inputSize = input.shape[1];
  if (weight.rank !== 2) {
    throw new Error(`The weight (rank ${weight.rank}) is not a 2-D tensor.`);
  }
  if (weight.shape[0] !== 3 * hiddenSize || weight.shape[1] !== inputSize) {
    throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]}] is invalid.`);
  }
  if (recurrentWeight.rank !== 2) {
    throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 2-D tensor.`);
  }
  if (recurrentWeight.shape[0] !== 3 * hiddenSize || recurrentWeight.shape[1] !== hiddenSize) {
    throw new Error(`The shape of recurrentWeight ` +
      `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}] is invalid.`);
  }
  if (hiddenState.rank !== 2) {
    throw new Error(`The hiddenState (rank ${hiddenState.rank}) is not a 2-D tensor.`);
  }
  if (hiddenState.shape[0] !== batchSize || hiddenState.shape[1] !== hiddenSize) {
    throw new Error(`The shape of hiddenState [${hiddenState.shape[0]}, 
    ${hiddenState.shape[1]}] is invalid.`);
  }
  if (bias) {
    if (bias.rank !== 1) {
      throw new Error(`The bias (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (bias.shape[0] !== 3 * hiddenSize) {
      throw new Error(`The shape of bias [${bias.shape[0]}] is invalid.`);
    }
  }
  if (recurrentBias) {
    if (recurrentBias.rank !== 1) {
      throw new Error(`The recurrentBias (rank ${bias.rank}) is not a 1-D tensor.`);
    }
    if (recurrentBias.shape[0] !== 3 * hiddenSize) {
      throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]}] is invalid.`);
    }
  }
  if (layout !== 'zrn' && layout !== 'rzn') {
    throw new Error(`The layout ${layout} is invalid.`);
  }
}

export function validateGruParams(input, weight, recurrentWeight, steps, hiddenSize,
    {bias, recurrentBias, initialHiddenState,
      direction = 'forward', layout = 'zrn'}) {
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new Error(`The steps ${steps} is invalid.`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
  }
  if (input.rank !== 3) {
    throw new Error(`The input (rank ${input.rank}) is not a 3-D tensor.`);
  }
  if (input.shape[0] !== steps) {
    throw new Error(`The input.shape[0] ${input.shape[0]} is not equal to steps ${steps}.`);
  }
  const batchSize = input.shape[1];
  const inputSize = input.shape[2];
  if (direction !== 'forward' && direction !== 'backward' && direction !== 'both') {
    throw new Error(`The direction ${direction} is invalid.`);
  }
  const numDirections = (direction === 'both' ? 2 : 1);
  if (weight.rank !== 3) {
    throw new Error(`The weight (rank ${weight.rank}) is not a 3-D tensor.`);
  }
  if (weight.shape[0] !== numDirections || weight.shape[1] !== 3 * hiddenSize ||
      weight.shape[2] !== inputSize) {
    throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]},
        ${weight.shape[2]}] is invalid.`);
  }
  if (recurrentWeight.rank !== 3) {
    throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 3-D tensor.`);
  }
  if (recurrentWeight.shape[0] !== numDirections ||
      recurrentWeight.shape[1] !== 3 * hiddenSize ||
      recurrentWeight.shape[2] !== hiddenSize) {
    throw new Error(`The shape of recurrentWeight ` +
                    `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}, ` +
                    `${recurrentWeight.shape[2]}] is invalid.`);
  }
  if (bias) {
    if (bias.rank !== 2) {
      throw new Error(`The bias (rank ${bias.rank}) is not a 2-D tensor.`);
    }
    if (bias.shape[0] !== numDirections || bias.shape[1] !== 3 * hiddenSize) {
      throw new Error(`The shape of bias [${bias.shape[0]}, ${bias.shape[1]}] is invalid.`);
    }
  }
  if (recurrentBias) {
    if (recurrentBias.rank !== 2) {
      throw new Error(`The recurrentBias (rank ${recurrentBias.rank}) is not a 2-D tensor.`);
    }
    if (recurrentBias.shape[0] !== numDirections || recurrentBias.shape[1] !== 3 * hiddenSize) {
      throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]},
          ${recurrentBias.shape[1]}] is invalid.`);
    }
  }
  if (initialHiddenState) {
    if (initialHiddenState.rank !== 3) {
      throw new Error(
          `The initialHiddenState (rank ${initialHiddenState.rank}) is not a 3-D tensor.`);
    }
    if (initialHiddenState.shape[0] !== numDirections ||
        initialHiddenState.shape[1] !== batchSize ||
        initialHiddenState.shape[2] !== hiddenSize) {
      throw new Error(`The shape of initialHiddenState [${initialHiddenState.shape[0]},
          ${initialHiddenState.shape[1]}, ${initialHiddenState.shape[2]}] is invalid.`);
    }
  }
  if (layout !== 'zrn' && layout !== 'rzn') {
    throw new Error(`The layout ${layout} is invalid.`);
  }
}

export function validateMatmulParams(a, b) {
  const aCols = a.shape[a.rank - 1];
  const bRows = b.shape[b.rank - 2];
  if (aCols !== bRows) {
    throw new Error(
        `The columns (${aCols}) of input a is not equal to rows (${bRows}) of input b.`);
  }
}

export function validatePool2dParams(input, _, {roundingType = 'floor'}) {
  if (input.rank !== 4) {
    throw new Error('The input should be a 4-D tensor.');
  }
  if (roundingType !== 'floor' && roundingType !== 'ceil') {
    throw new Error('The rounding type is invalid.');
  }
}

export function validateReduceParams(input, {axes}) {
  if (axes === undefined) {
    return;
  }
  if (axes.length > input.rank) {
    throw new Error(`The length ${axes.length} of axes is bigger` +
                    `than input rank ${input.rank}.`);
  }
  for (let i = 0; i < axes.length; ++i) {
    if (axes[i] < 0 || axes[i] >= input.rank) {
      throw new Error(`The value ${axes[i]} at axis ${i} of axes is invalid.`);
    }
  }
}

export function validateSliceParams(input, starts, sizes) {
  const rank = input.rank;
  if (starts.length !== rank) {
    throw new Error(`The length ${starts.length} of starts is not equal to the length ` +
                    `${rank} of input.`);
  }
  if (sizes.length !== rank) {
    throw new Error(`The length ${sizes.length} of sizes is not equal` +
                    ` to the length ${rank} of input.`);
  }
  for (let i = 0; i < rank; ++i) {
    const size = input.shape[i];
    const start = starts[i];
    if (!Number.isInteger(start) || start < 0 ) {
      throw new Error(`Invalid starts value ${start} - it should be an unsigned integer.`);
    }
    if (start >= size) {
      throw new Error(`Invalid starts value ${start} - it shoule be in the interval ` +
                      `[0, ${size}).`);
    } else {
      const sliceSize = sizes[i];
      if (!Number.isInteger(sliceSize) || sliceSize <= 0) {
        throw new Error(`Invalid sizes value ${sliceSize} - it should be an unsigned integer.`);
      }
      if (start + sliceSize > size) {
        throw new Error(`Invalid sizes value ${sliceSize} - the sum of the start ${start} ` +
          `plus the size ${sliceSize} is greater than the dimensional size ${size}`);
      }
    }
  }
}

export function validateSoftmaxParams(input, axis) {
  const rank = input.rank;
  if (!Number.isInteger(axis) || axis < 0) {
    throw new Error(`The axis ${axis} should be an unsigned integer.`);
  }
  if (axis >= rank) {
    throw new Error(`The axis ${axis} should be in the interval [0, ${rank}).`);
  }
}

export function validateSplitParams(input, splits, {axis = 0} = {}) {
  if (axis !== undefined) {
    const rank = input.rank;
    if (!Number.isInteger(axis) || axis < 0) {
      throw new Error(`The axis ${axis} should be an unsigned integer.`);
    }
    if (axis >= rank) {
      throw new Error(`The axis ${axis} should be in the interval [0, ${rank}).`);
    }
  }
  if (typeof splits === 'number') {
    if (!Number.isInteger(splits) || splits <= 0) {
      throw new Error(`Invalid splits ${splits} - it should be a positive integer.`);
    }
    if (input.shape[axis] % splits !== 0) {
      throw new Error(`The splits ${splits} must evenly divide the dimension size ` +
                      `${input.shape[axis]} of input along options.axis ${axis}.`);
    }
  } else if (splits instanceof Array) {
    if (!splits.every((v) => Number.isInteger(v) && v > 0)) {
      throw new Error(`Invalid splits ${splits} - it should be an Array of positive integers.`);
    }
    const sum = splits.reduce((a, b) => a + b);
    if (sum !== input.shape[axis]) {
      throw new Error(`Invalid [${splits}] - the sum of sizes ${sum} must equal ` +
                      `to the dimension size ${input.shape[axis]} of input` +
                      ` along options.axis ${axis}`);
    }
  }
}

export function validateSqueezeParams(input, {axes} = {}) {
  if (axes) {
    if (axes.length > input.rank) {
      throw new Error(`The length of axes ${axes.length} is bigger ` +
                      `than input rank ${input.rank}.`);
    }

    for (const axis of axes) {
      if (axis < 0 || axis >= input.rank) {
        throw new Error(`The value of axes ${axis} is invalid.`);
      }
      if (axes && input.shape[axis] !== 1) {
        throw new Error(`The value ${input.shape[axis]} ` +
                        `at axis ${axis} of input shape is not 1.`);
      }
    }
  }
}

export function validateTranposeParams(input, {permutation}) {
  if (permutation.length !== input.rank) {
    throw new Error(
        `The permutation length ${permutation.length} is not equal to rank ${input.rank}.`);
  }
}

export function validateNotParams(input) {
  for (let i = 0; i < sizeOfShape(input.shape); ++i) {
    const a = input.getValueByIndex(i);
    if (!Number.isInteger(a) || a < 0 || a > 255) {
      throw new Error('Invalid input value - it should be an integer in the interval [0, 255]');
    }
  }
}

export function validateGatherParams(input, indices, {axis = 0} = {}) {
  if (axis !== undefined) {
    const rank = input.rank;
    if (!Number.isInteger(axis) || axis < 0) {
      throw new Error(`The axis ${axis} should be an unsigned integer.`);
    }
    if (axis >= rank) {
      throw new Error(`The axis ${axis} should be in the interval [0, ${rank}).`);
    }
  }
  const axisSize = input.shape[axis];
  for (let i = 0; i < sizeOfShape(indices.shape); ++i) {
    const index = indices.getValueByIndex(i);
    if (!Number.isInteger(index) || index < -axisSize || index >= axisSize) {
      throw new Error(`Invalid indices value - it should be an integer in the interval ` +
          `[${-axisSize}, ${axisSize - 1}]`);
    }
  }
}

export function validateScatterElementsParams(input, indices, updates, {axis = 0} = {}) {
  const inputRank = input.rank;
  const indicesRank = indices.rank;
  const updatesRank = updates.rank;
  if (inputRank < 1) {
    throw new Error(`The input should be at least a 1-D tensor.`);
  }
  if (indicesRank < 1) {
    throw new Error(`The indices should be at least a 1-D tensor.`);
  }
  if (updatesRank < 1) {
    throw new Error(`The updates should be at least a 1-D tensor.`);
  }
  if (indicesRank !== inputRank) {
    throw new Error(`Invalid indices value - indices rank should be equal to input rank.`);
  }
  if (updatesRank !== indicesRank) {
    throw new Error(`Invalid updates value - updates rank should be equal to indices rank.`);
  }
  if (!updates.shape.every((value, index) => value === indices.shape[index])) {
    throw new Error(`Invalid updates value - updates shape should be same as indices shape.`);
  }
  if (!Number.isInteger(axis) || axis < 0 || axis >= inputRank) {
    throw new Error(
        `The axis ${axis} should be an unsigned integer in the interval [0, ${inputRank}).`);
  }
  const axisSize = input.shape[axis];
  for (let i = 0; i < sizeOfShape(indices.shape); ++i) {
    const index = indices.getValueByIndex(i);
    if (!Number.isInteger(index) || index < -axisSize || index >= axisSize) {
      throw new Error(`Invalid indices value - it should be an integer in the interval ` +
      `[${-axisSize}, ${axisSize - 1}]`);
    }
  }
}

export function validateGatherElementsParams(input, indices, {axis = 0} = {}) {
  const inputRank = input.rank;
  const indicesRank = indices.rank;
  if (inputRank < 1) {
    throw new Error(`The input should be at least a 1-D tensor.`);
  }
  if (indicesRank < 1) {
    throw new Error(`The indices should be at least a 1-D tensor.`);
  }
  if (indicesRank !== inputRank) {
    throw new Error(`Invalid indices value - indices rank should be equal to input rank`);
  }
  if (!Number.isInteger(axis) || axis < 0 || axis >= inputRank) {
    throw new Error(
        `The axis ${axis} should be an unsigned integer in the interval [0, ${inputRank}).`);
  }
  const axisSize = input.shape[axis];
  for (let i = 0; i < sizeOfShape(indices.shape); ++i) {
    const index = indices.getValueByIndex(i);
    if (!Number.isInteger(index) || index < -axisSize || index >= axisSize) {
      throw new Error(`Invalid indices value - it should be an integer in the interval ` +
          `[${-axisSize}, ${axisSize - 1}]`);
    }
  }
}

export function validateCumulativeSumParams(input, axis) {
  if (axis !== undefined) {
    const rank = input.rank;
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank) {
      throw new Error(`The axis ${axis} must be an unsigned integer in the interval [0, ${rank}).`);
    }
  }
}

export function validateTriangularParams(input, {diagonal = 0} = {}) {
  const inputRank = input.rank;
  if (inputRank < 2) {
    throw new Error('The input should be at least a 2-D tensor.');
  }
  if (!Number.isInteger(diagonal)) {
    throw new Error(`The diagonal should be an integer.`);
  }
}

export function validateTileParams(input, repetitions) {
  const inputRank = input.rank;
  const repetitionsLength = repetitions.length;
  if (repetitionsLength != inputRank ) {
    throw new Error(
        `The repetitions length ${repetitionsLength} is not equal to rank ${inputRank}.`);
  }
  if (!repetitions.every((v) => Number.isInteger(v) && v > 0)) {
    throw new Error(
        `Invalid repetitions ${repetitions} - it should be an Array of positive integers.`);
  }
}
