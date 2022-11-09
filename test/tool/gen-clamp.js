'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {clamp} from '../../src/clamp.js';
import {Tensor} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function clampCompute(inputShape, inputValue, options = {}) {
    const inputTensor = new Tensor(inputShape, inputValue);
    const outputTensor = clamp(inputTensor, options);
    return outputTensor.data;
  }

  const min = -10.0;
  const max = 10.0;
  const savedDataFile = path.join(path.dirname(process.argv[1]), 'test_data', 'clamp_data.json');
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  // use toSaveDataDict variable to save inputs data which are new generated or
  // already recorded in existed test data file
  const toSaveDataDict = utils.prepareInputsData(inputsDataInfo, savedDataFile, min, max);
  // use toSaveDataDict variable to also save expected data genertated by clamp tests
  // with above input data
  toSaveDataDict['expectedData'] = {};
  const tests = jsonDict.tests;

  for (const test of tests) {
    const expectedDataCategory = test.expected.data;
    if (toSaveDataDict['expectedData'][expectedDataCategory] === undefined) {
      const precisionType = test.type;
      const inputShape = test.input.shape;
      const inputDataCategory = test.input.data;
      const precisionData = utils.getPrecisionDataFromDataDict(
          toSaveDataDict['inputsData'], inputDataCategory, precisionType);
      const options = test.options;
      const result = clampCompute(inputShape, precisionData, options);
      toSaveDataDict['expectedData'][expectedDataCategory] =
          utils.getPrecisionData(result, precisionType);
    } else {
      continue;
    }
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data onto ${savedDataFile} .`);
})();
