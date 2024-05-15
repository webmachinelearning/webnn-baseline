'use strict';

import fs from 'fs';
import path from 'path';
import {Float16Array} from '@petamoriken/float16';

import {Tensor, sizeOfShape} from '../../src/lib/tensor.js';
import {neg} from '../../src/unary.js';
import {transpose} from '../../src/transpose.js';


/**
 * Convert data as required precision type.
 * @param {Array<Number>} input
 * @param {String} precisionType
 * @return {Array<Number>}
 */
function getPrecisionData(input, precisionType) {
  let data;
  const isNumber = typeof input === 'number';
  if (isNumber) {
    input = [input];
  }
  switch (precisionType) {
    case 'float32':
      data = new Float32Array(input);
      break;
    case 'float16':
      data = new Float16Array(input);
      break;
    case 'float64':
      data = new Float64Array(input);
      break;
    case 'uint8':
      data = new Uint8Array(input);
      break;
    case 'uint32':
      data = new Uint32Array(input);
      break;
    case 'int64':
      // data = new BigInt64Array(input.map(x => BigInt(x)));
      // data = new Array(input.map(x => BigInt(x).toString()));
      data = new Int32Array(input);
      break;
    default:
      break;
  }
  if (isNumber) {
    data = data[0];
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
 * @param {String} type default 'float64'
 * @param {String} sign default 'positive'
 * @return {Number}
 */
function getRandom(min, max, type = 'float64', sign = 'positive') {
  if (sign === 'positive') {
    if (min < 0) {
      min = 0;
    }
  } else if (sign === 'negative') {
    if (max > 0) {
      max = 0;
    }
  }

  if (type === 'float64' || type === 'float32' || type === 'float16') {
    return Math.random() * (max - min) + min;
  } else if (type === 'int32') {
    min = Math.ceil(min) + 1;
    max = Math.floor(max) - 1;
    // The maximum is exclusive and the minimum is inclusive
    return Math.floor(Math.random() * (max - min) + min);
  } else if (type === 'int64'||type === 'uint32') {
    return Math.floor(Math.random() * (max - min + 1) + min);
  } else if (type === 'uint8') {
    let randomUint8Value;
    if (Math.random() < 0.2) {
      randomUint8Value = 0;
    } else {
      randomUint8Value = Math.floor(Math.random() * 255);
    }
    return randomUint8Value;
  }
}

/**
 * Get random numbers between the specified values
 * @param {Number} min
 * @param {Number} max
 * @param {Number} size
 * @param {String} [type='float64']
 * @param {String} [sign='mixed']
 * @return {Array<Number>}
 */
function getRandomNumbers(min, max, size, type = 'float64', sign='mixed') {
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = getRandom(min, max, type, sign);
  }
  return getPrecisionData(data, type);
}

/**
 * Prepare input data by specified config of inputsDataInfo, dataFile, min,
 * max parameters .
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
        const srcDataInfo = inputsDataInfo[targetDataInfo.data];
        const inputTensor = new Tensor(
            srcDataInfo.shape, dstDataDict['inputsData'][targetDataInfo.data]);
        let outputTensor;
        if (targetDataInfo.processCategory === 'transpose') {
          outputTensor = transpose(
              inputTensor, {permutation: targetDataInfo.permutation});
        } else if (targetDataInfo.processCategory === 'negative') {
          outputTensor = neg(inputTensor);
        }
        console.log(`source ${source}`);
        dstDataDict['inputsData'][source] = outputTensor.data;
      } else {
        const total = sizeOfShape(targetDataInfo.shape);
        const sign = targetDataInfo.sign;
        const type = targetDataInfo.type;
        if (targetDataInfo.dataRange) {
          min = targetDataInfo.dataRange[0];
          max = targetDataInfo.dataRange[1];
        }
        const generatedNumbers =
            utils.getRandomNumbers(min, max, total, type, sign);
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
      content.replace(/\\"|"(?:\\"|[^"])*"|(\/\/.*|\/\*[\s\S]*?\*\/)/g,
          (m, g) => g ? '' : m));
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
         value instanceof Float16Array ||
         value instanceof Float64Array ) {
      if (value.length ===1) { // value instanceof Uint8Array &&
        const result = [];
        result[0] = value[0];
        return result;
      } else {
        return Array.apply([], value);
      }
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
