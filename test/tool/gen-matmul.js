'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {matmul} from '../../src/matmul.js';
import {Scalar, Tensor} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function matmulCompute(inputA, inputB) {
    const inputTensorA = new Tensor(inputA.shape, inputA.data);
    const inputTensorB = new Tensor(inputB.shape, inputB.data);
    const outputTensor = matmul(inputTensorA, inputTensorB);
    let result = outputTensor.data;
    if (outputTensor instanceof Scalar) {
      // scalar
      result = result[0];
    }
    return result;
  }

  const min = -100.0;
  const max = 100.0;
  const savedDataFile = path.join(path.dirname(process.argv[1]), 'test-data', 'matmul-data.json');
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  // use toSaveDataDict variable to save inputs data which are new generated or
  // already recorded in existed test data file
  const toSaveDataDict = utils.prepareInputsData(inputsDataInfo, savedDataFile, min, max);
  // use toSaveDataDict variable to also save expected data genertated by matmul tests
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
    const precisionDataA = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], test.inputs.a.data, precisionType);
    const inputA = {shape: test.inputs.a.shape, data: precisionDataA};
    const precisionDataB = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], test.inputs.b.data, precisionType);
    const inputB = {shape: test.inputs.b.shape, data: precisionDataB};
    const result = matmulCompute(inputA, inputB);
    toSaveDataDict['expectedData'][expectedDataSource] =
      utils.getPrecisionData(result, precisionType);
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data onto ${savedDataFile} .`);
})();
