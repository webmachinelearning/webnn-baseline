'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {concat} from '../../src/concat.js';
import {Tensor, sizeOfShape} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function concatCompute(inputShapeValues, axis) {
    const inputs = [];
    for (let i = 0; i < inputShapeValues.length; i++) {
      inputs.push(new Tensor(inputShapeValues[i].shape, inputShapeValues[i].data));
    }
    const outputTensor = concat(inputs, axis);
    return outputTensor.data;
  }

  const min = -1.0;
  const max = 1.0;
  const savedDataFile = path.join(path.dirname(process.argv[1]), 'test-data', 'concat-data.json');
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  // use toSaveDataDict variable to save inputs data which are new generated or
  // already recorded in existed test data file
  const toSaveDataDict = utils.prepareInputsData(inputsDataInfo, savedDataFile, min, max);
  // use toSaveDataDict variable to also save expected data genertated by concat tests
  // with above input data
  toSaveDataDict['expectedData'] = {};
  const tests = jsonDict.tests;

  for (const test of tests) {
    const expectedDataSource = test.expected.data;
    if (toSaveDataDict['expectedData'][expectedDataSource] !== undefined) {
      // The expected data is already defined.
      continue;
    }
    const axis = test.axis;
    const precisionType = test.type;
    const inputShapes = test.inputs.shape;
    const inputDataCategory = test.inputs.data;
    const feedData = toSaveDataDict['inputsData'][inputDataCategory];
    const inputShapeValues = [];
    let position = 0;
    for (const shape of inputShapes) {
      const size = sizeOfShape(shape);
      const precisionData =
          utils.getPrecisionData(feedData.slice(position, position + size), precisionType);
      inputShapeValues.push({shape, data: precisionData});
      position += size;
    }
    const result = concatCompute(inputShapeValues, axis);
    toSaveDataDict['expectedData'][expectedDataSource] =
        utils.getPrecisionData(result, precisionType);
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data onto ${savedDataFile} .`);
})();
