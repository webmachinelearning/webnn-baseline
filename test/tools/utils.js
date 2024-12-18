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
 * @return {(Array<Number>|Number)}
 */
function getPrecisionData(input, precisionType) {
  let data;
  const isNumber = typeof input === 'number';
  if (isNumber) {
    input = [input];
  }

  switch (precisionType) {
    case 'float16':
      data = new Float16Array(input);
      break;
    case 'float32':
      data = new Float32Array(input);
      break;
    case 'int8':
      data = new Int8Array(input);
      break;
    case 'uint8':
      data = new Uint8Array(input);
      break;
    case 'int32':
      data = new Int32Array(input);
      break;
    case 'uint32':
      data = new Uint32Array(input);
      break;
    case 'int64':
      data = new BigInt64Array(input.map((x) => BigInt(x)));
      break;
    case 'uint64':
      data = new BigUint64Array(input.map((x) => BigInt(x)));
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
 * @return {(Array<Number>|Number)}
 */
function getPrecisionDataFromDataDict(srcDataDict, source, precisionType) {
  const feedData = srcDataDict[source];
  return getPrecisionData(feedData, precisionType);
}

/**
 * Get a random number by specified dataRange and dataType.
 * @param {{min: Number, max: Number, sign: String}} dataRange
 * @param {String} dataType
 * @return {Number}
 */
function getRandom(dataRange, dataType) {
  function getFloatRandomInclusive() {
    // The Math. random() method returns a random floating point number between 0 (inclusive)
    // and 1 (exclusive).
    const value = Math.min(Math.random() * 2, 1);
    return value;
  }

  function validateMinMaxByType(min, max, type) {
    if (type === 'int8' && (min < -128 || max > 127)) {
      throw new Error(`The range of int8 type should be [-128, 127].`);
    } else if (type === 'uint8' && (min < 0 || max > 255)) {
      throw new Error(`The range of uint8 type should be [0, 255].`);
    } else if (type === 'int32' && (min < -Math.pow(2, 31) || max > Math.pow(2, 31) - 1)) {
      throw new Error(
          `The range of int32 type should be [${-Math.pow(2, 31)}, ${Math.pow(2, 31) - 1}].`);
    } else if (type === 'uint32' && (min < 0 || max > Math.pow(2, 32) - 1)) {
      throw new Error(`The range of uint32 type should be [0, ${Math.pow(2, 32) - 1}].`);
    } else if (type === 'float16') {
      const fp16Max = (2 - Math.pow(2, -10)) * Math.pow(2, 15);
      if (min < -fp16Max || max > fp16Max) {
        throw new Error(`The range of float16 type should be [${-fp16Max}, ${fp16Max}].`);
      }
    } else if (type === 'int64') {
      const int64Max = 2n ** 64n - 1n;
      if (min < -int64Max - 1n || max > int64Max) {
        throw new Error('The range of int64 type should be ' +
          `[${-int64Max - 1n}, ${int64Max}].`);
      }
    } else if (type === 'uint64') {
      // In JavaScript, you can represent a BigUint64 using the BigInt data type.
      const int64Max = 2n ** 64n - 1n;
      if (min < 0n || max > int64Max) {
        throw new Error(`The range of uint64 type should be [0n, ${int64Max}].`);
      }
    } else if (type === 'float32') {
      const fp32Max = (2 - Math.pow(2, -23)) * Math.pow(2, 127);
      if (min < -fp32Max || max > fp32Max) {
        throw new Error(`The range of float32 type should be [${-fp32Max}, ${fp32Max}].`);
      }
    }
  }

  let min = dataRange.min;
  let max = dataRange.max;
  if (min > max) {
    throw new Error(`The min should be lesser than max.`);
  }

  validateMinMaxByType(min, max, dataType);

  const sign = dataRange.sign || 'mixed';
  if (sign === 'positive') {
    if (max <= 0) {
      throw new Error(`The max should be greater than 0 when sign is set as 'positive'.`);
    }
    if (min < 0) {
      min = 0;
    }
  } else if (sign === 'negative') {
    if (min >= 0) {
      throw new Error(`The min should be lesser than 0 when sign is set as 'negative'.`);
    }
    if (max > 0) {
      max = 0;
    }
  } else {
    // No change on min and max for mixed sign
  }

  const factor = getFloatRandomInclusive();
  let data;
  if (dataType === 'float32' || dataType === 'float16') {
    data = factor * (max - min) + min;
  } else if (!['int64', 'uint64'].includes(dataType)) {
    // integer
    const minCeiled = Math.ceil(min);
    const maxFloored = Math.floor(max);
    data = Math.floor(factor * (maxFloored - minCeiled) + minCeiled);
  } else {
    const convertedMin = Math.ceil(parseInt(min));
    const convertedMax = Math.floor(parseInt(max));
    data = BigInt(Math.floor(factor * (convertedMax - convertedMin) + convertedMin));
  }

  return data;
}

/**
 * Get random numbers of TypedArray.
 * @param {Number} size
 * @param {{min: Number, max: Number, sign: String}} dataRange
 * @param {String} dataType
 * @return {(Array<Number>|Number)}
 */
function getRandomNumbers(size, dataRange, dataType) {
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = getRandom(dataRange, dataType);
  }
  return getPrecisionData(data, dataType);
}

/**
 * Prepare input data by specified config of inputsDataInfo, dataFile, min,
 * max parameters.
 * @param {Object} inputsDataInfo information object for input data
 * @param {String} dataFile saved data file path
 * @param {{min: Number, max: Number}} dataRange
 * @return {Object}
 */
function prepareInputsData(inputsDataInfo, dataFile, dataRange) {
  const dstDataDict = {inputsData: {}};
  let srcDataDict = {};
  if (fs.existsSync(dataFile)) {
    srcDataDict = readJsonFile(dataFile);
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
        dstDataDict['inputsData'][source] = outputTensor.data;
      } else {
        const total = sizeOfShape(targetDataInfo.shape);
        if (targetDataInfo.dataRange !== undefined) {
          // Specified data range
          dataRange = targetDataInfo.dataRange;
        }
        const generatedNumbers = getRandomNumbers(total, dataRange, targetDataInfo.dataType);
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
      content.replace(/\\"|"(?:\\"|[^"])*"|(\/\/.*|\/\*[\s\S]*?\*\/)/g, // remove comments
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
         value instanceof Int32Array ||
         value instanceof Uint32Array ||
         value instanceof BigInt64Array ||
         value instanceof BigUint64Array ||
         value instanceof Float16Array ||
         value instanceof Float32Array) {
      if (value.length === 1) {
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
