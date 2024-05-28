'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {softsign} from '../../src/softsign.js';
import {gelu} from '../../src/gelu.js';
import {Tensor} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function computeBySingleInput(operatorName, input, options = {}) {
    const operatorMappingDict = {
      'gelu': gelu,
      'softsign': softsign,
    };
    const inputTensor = new Tensor(input.shape, input.data);
    const outputTensor =
        operatorMappingDict[operatorName](inputTensor, options);
    return outputTensor.data;
  }

  const testDataFileName = path.basename(process.argv[2]);
  const operatorString =
      testDataFileName.slice(0, testDataFileName.indexOf('.json'));
  const savedDataFile = path.join(
      path.dirname(process.argv[1]), 'test-data',
      `${operatorString}-data.json`);
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  const inputsDataRange = jsonDict.inputsDataRange;
  const toSaveDataDict = utils.prepareInputsData(
      inputsDataInfo, savedDataFile, inputsDataRange.min, inputsDataRange.max);
  toSaveDataDict['expectedData'] = {};
  const tests = jsonDict.tests;
  const wptTests = JSON.parse(JSON.stringify(tests));
  for (const test of tests) {
    console.log(`name ${test.name}`);
    const precisionDataInput = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], test.inputs.input.data,
        test.inputs.input.type);
    const input = {shape: test.inputs.input.shape, data: precisionDataInput};
    const result = computeBySingleInput(operatorString, input, test.options);
    toSaveDataDict['expectedData'][test.expected.data] =
      utils.getPrecisionData(result, test.expected.type);
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data into ${savedDataFile}.`);

  const wptConformanceTestsDict = {tests: []};
  for (const test of wptTests) {
    // update inputs data
    for (const inputName in test.inputs) {
      test.inputs[inputName].data =
          typeof test.inputs[inputName].data === 'number' ||
          (typeof test.inputs[inputName].data === 'object' &&
           typeof test.inputs[inputName].data[0] === 'number') ?
          test.inputs[inputName].data :
          utils.getPrecisionDataFromDataDict(
              toSaveDataDict['inputsData'], test.inputs[inputName].data,
              test.inputs[inputName].type);
    }
    // update weights (scale, bias, and etc.) data of options
    if (test.options) {
      for (const optionName in test.options) {
        if (test.options[optionName].data) {
          test.options[optionName].data =
              toSaveDataDict['inputsData'][test.options[optionName].data];
        }
      }
    }
    // update expected data
    test.expected.data = toSaveDataDict['expectedData'][test.expected.data];
    wptConformanceTestsDict.tests.push(test);
  }
  const savedWPTDataFile = path.join(
      path.dirname(process.argv[1]), 'test-data-wpt', `${operatorString}.json`);
  utils.writeJsonFile(wptConformanceTestsDict, savedWPTDataFile);

  console.log(`[ Done ] Generate test data file for WPT tests.`);
})();
