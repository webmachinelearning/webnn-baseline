'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {gemm} from '../../src/gemm.js';
import {Tensor} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function gemmCompute(a, b, options = {}) {
    const inputA = new Tensor(a.shape, a.data);
    const inputB = new Tensor(b.shape, b.data);
    const outputTensor = gemm(inputA, inputB, options);
    return outputTensor.data;
  }

  const min = -100.0;
  const max = 100.0;
  const savedDataFile = path.join(path.dirname(process.argv[1]), 'test-data', 'gemm-data.json');
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  // use toSaveDataDict variable to save inputs data which are new generated or
  // already recorded in existed test data file
  const toSaveDataDict = utils.prepareInputsData(inputsDataInfo, savedDataFile, min, max);
  // use toSaveDataDict variable to also save expected data genertated by gemm tests
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
    const a = {shape: test.inputs.a.shape, data: precisionDataA};
    const precisionDataB = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], test.inputs.b.data, precisionType);
    const b = {shape: test.inputs.b.shape, data: precisionDataB};
    const options = test.options;
    const gemmOptions = {};
    if (options !== undefined) {
      if (options.c !== undefined) {
        const precisionDataC = utils.getPrecisionDataFromDataDict(
            toSaveDataDict['inputsData'], options.c.data, precisionType);
        if (options.c.shape !== undefined) {
          gemmOptions.c = new Tensor(options.c.shape, precisionDataC);
        } else {
          gemmOptions.c = precisionDataC[0];
        }
      }
      if (options.alpha !== undefined) {
        const precisionDataAlpha = utils.getPrecisionDataFromDataDict(
            toSaveDataDict['inputsData'],
            options.alpha.data,
            precisionType);
        gemmOptions.alpha = precisionDataAlpha[0];
      }
      if (options.beta !== undefined) {
        const precisionDataBeta = utils.getPrecisionDataFromDataDict(
            toSaveDataDict['inputsData'],
            options.beta.data,
            precisionType);
        gemmOptions.beta = precisionDataBeta[0];
      }
      if (options.aTranspose !== undefined) {
        gemmOptions.aTranspose = options.aTranspose;
      }
      if (options.bTranspose !== undefined) {
        gemmOptions.bTranspose = options.bTranspose;
      }
    }
    const result = gemmCompute(a, b, gemmOptions);
    toSaveDataDict['expectedData'][expectedDataSource] =
        utils.getPrecisionData(result, precisionType);
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data onto ${savedDataFile} .`);
})();
