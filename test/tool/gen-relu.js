'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {relu} from '../../src/relu.js';
import {Tensor} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function reluCompute(inputShape, inputValue) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = relu(inputTensor);
    return outputTensor.data;
  }

  const min = -100.0;
  const max = 100.0;
  const savedDataFile = path.join(path.dirname(process.argv[1]), 'test-data', 'relu-data.json');
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  // use toSaveDataDict variable to save inputs data which are new generated or
  // already recorded in existed test data file
  const toSaveDataDict = utils.prepareInputsData(inputsDataInfo, savedDataFile, min, max);
  // use toSaveDataDict variable to also save expected data genertated by relu tests
  // with above input data
  toSaveDataDict['expectedData'] = {};
  const tests = jsonDict.tests;

  for (const test of tests) {
    const expectedDataSource = test.expected.data;
    if (toSaveDataDict['expectedData'][expectedDataSource] !== undefined) {
      // The expected data is already defined.
      continue;
    }
    const precisionType = test.type;
    const inputShape = test.input.shape;
    const inputDataSource = test.input.data;
    const precisionData = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], inputDataSource, precisionType);
    const result = reluCompute(inputShape, precisionData);
    toSaveDataDict['expectedData'][expectedDataSource] =
        utils.getPrecisionData(result, precisionType);
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data onto ${savedDataFile} .`);
})();
