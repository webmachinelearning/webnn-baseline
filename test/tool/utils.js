'use strict';

import fs from 'fs';
import path from 'path';
import {Tensor, sizeOfShape} from '../../src/lib/tensor.js';
import {transpose} from '../../src/transpose.js';


const getRandomFunctions = {
  float64: getRandomFloat64,
};

/**
 * Convert data as required precision type.
 * @param {Array<Number>} input
 * @param {String} precisionType
 * @return {Array<Number>}
 */
function getPrecisionData(input, precisionType) {
  let data;
  switch (precisionType) {
    case 'float32':
      data = new Float32Array(input);
      break;
    case 'float64':
      data = new Float64Array(input);
      break;
    // TODO: float16
    default:
      break;
  }
  return data;
}


/**
 * Get converted data from given data dict with specified field and precision type.
 * @param {Object} srcDataDict
 * @param {String} source
 * @param {String} precisionType
 * @return {Array<Number>}
 */
function getPrecisionDataFromDataDict(srcDataDict, source, precisionType) {
  const feedData = srcDataDict[source];
  return getPrecisionData(feedData, precisionType);
}

/**
 * Get a random number between the specified values
 * @param {Number} min
 * @param {Number} max
 * @return {Number}
 */
function getRandomFloat64(min, max) {
  return Math.random() * (max - min) + min;
}

/**
 * Get random numbers between the specified values
 * @param {Number} min
 * @param {Number} max
 * @param {Number} size
 * @param {String} [type='float64']
 * @return {Array<Number>}
 */
function getRandomNumbers(min, max, size, type = 'float64') {
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = getRandomFunctions[type](min, max);
  }
  return data;
}

/**
 * Prepare input data by specified config of inputsDataInfo, dataFile, min, max parameters .
 * @param {Object} inputsDataInfo information object for input data
 * @param {String} dataFile saved data file path
 * @param {Number} min
 * @param {Number} max
 * @return {Object}
 */
function prepareInputsData(inputsDataInfo, dataFile, min, max) {
  const dstDataDict = {inputsData: {}};
  let srcDataDict = {};
  if (fs.existsSync(dataFile)) {
    srcDataDict = utils.readJsonFile(dataFile);
  }
  for (const source in inputsDataInfo) {
    // reserve last input data when generating new required input data
    if (srcDataDict['inputsData'] !== undefined &&
        srcDataDict['inputsData'][source] !== undefined) {
      dstDataDict['inputsData'][source] = srcDataDict['inputsData'][source];
    } else {
      const targetDataInfo = inputsDataInfo[source];
      if (targetDataInfo.data !== undefined) {
        const permutation = targetDataInfo.transpose;
        const srcDataInfo = inputsDataInfo[targetDataInfo.data];
        const inputTensor =
          new Tensor(srcDataInfo.shape, dstDataDict['inputsData'][targetDataInfo.data]);
        const outputTensor = transpose(inputTensor, {permutation});
        dstDataDict['inputsData'][source] = outputTensor.data;
      } else {
        const total = sizeOfShape(targetDataInfo.shape);
        const type = targetDataInfo.type;
        const generatedNumbers = utils.getRandomNumbers(min, max, total, type);
        dstDataDict['inputsData'][source] = generatedNumbers;
      }
    }
  }
  return dstDataDict;
}

/**
 * Get JSON object from specified JSON file.
 * @param {String} filePath
 * @return {Object}
 */
function readJsonFile(filePath) {
  let inputFile;
  if (path.isAbsolute(filePath)) {
    inputFile = filePath;
  } else {
    inputFile =
        path.join(path.dirname(process.argv[1]), filePath);
  }
  const content = fs.readFileSync(inputFile).toString();
  const jsonDict = JSON.parse(
      content.replace(/\\"|"(?:\\"|[^"])*"|(\/\/.*|\/\*[\s\S]*?\*\/)/g, (m, g) => g ? '' : m));
  return jsonDict;
}

/**
 * Save JSON infomation into file.
 * @param {Object} jsonDict
 * @param {String} saveFile
 */
function writeJsonFile(jsonDict, saveFile) {
  const parentDirectory = path.dirname(saveFile);
  if (!fs.existsSync(parentDirectory)) {
    fs.mkdirSync(parentDirectory);
  }
  const jsonString = JSON.stringify(jsonDict, function(key, value) {
    // the replacer function is looking for some typed arrays.
    // If found, it replaces it by a trio
    if ( value instanceof Int8Array ||
         value instanceof Uint8Array ||
         value instanceof Uint16Array ||
         value instanceof Int32Array ||
         value instanceof Uint32Array ||
         value instanceof Float32Array ||
         value instanceof Float64Array ) {
      return Array.apply([], value);
    }
    return value;
  }, 2);
  fs.writeFileSync(saveFile, jsonString);
}

export const utils = {
  getPrecisionData: getPrecisionData,
  getPrecisionDataFromDataDict: getPrecisionDataFromDataDict,
  getRandomNumbers: getRandomNumbers,
  prepareInputsData: prepareInputsData,
  readJsonFile: readJsonFile,
  writeJsonFile: writeJsonFile,
};
