'use strict';

/* eslint guard-for-in: 0 */

import fs from 'fs';
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

  const savedDataFile = path.join(path.dirname(process.argv[1]), 'test_data', 'concat_data.json');
  let savedDataDict = {};
  if (fs.existsSync(savedDataFile)) {
    savedDataDict = utils.readJsonFile(savedDataFile);
  }

  // use toSaveDataDict var to save inputs data and expected data
  const toSaveDataDict = {};
  // Step1: Genertate random input data
  toSaveDataDict['inputsData'] = {};
  const min = -1.0;
  const max = 1.0;
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsData = jsonDict.inputsData;
  for (const category in inputsData) {
    // reserve last input data when generating new required input data
    if (savedDataDict['inputsData'] !== undefined &&
        savedDataDict['inputsData'][category] !== undefined) {
      toSaveDataDict['inputsData'][category] = savedDataDict['inputsData'][category];
    } else {
      const targetDataInfo = inputsData[category];
      const total = sizeOfShape(targetDataInfo.shape);
      const type = targetDataInfo.type;
      const geneartedNumbers = utils.getRandomNumbers(min, max, total, type);
      toSaveDataDict['inputsData'][category] = geneartedNumbers;
    }
  }

  // Step2: Genertate expected data by concat tests with generated input data
  toSaveDataDict['expectedData'] = {};
  const tests = jsonDict.tests;

  for (const test of tests) {
    const expectedDataCategory = test.expected.data;
    if (toSaveDataDict['expectedData'][expectedDataCategory] === undefined) {
      const axis = test.axis;
      const precisionType = test.type;
      const inputShapes = test.inputs.shape;
      const inputDataCategory = test.inputs.data;
      const feedData = toSaveDataDict['inputsData'][inputDataCategory];
      const inputShapeValues = [];
      let pos = 0;
      for (const shape of inputShapes) {
        const size = sizeOfShape(shape);
        const precisionData =
            utils.getPrecisionData(feedData.slice(pos, pos + size), precisionType);
        inputShapeValues.push({shape, data: precisionData});
        pos += size;
      }
      const result = concatCompute(inputShapeValues, axis);
      toSaveDataDict['expectedData'][expectedDataCategory] =
          utils.getPrecisionData(result, precisionType);
    } else {
      continue;
    }
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data onto ${savedDataFile} .`);
})();
