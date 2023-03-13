/**
 * Compute the beginning and ending pad given input, filter and stride.
 * @param {String} autoPad
 * @param {Number} inputSize
 * @param {Number} effectiveFilterSize
 * @param {Number} stride
 * @param {Number} outputPadding
 * @return {Array} [paddingBegin, paddingEnd]
 */
export function computePaddingForAutoPad(
    autoPad, inputSize, effectiveFilterSize, stride, outputPadding) {
  let totalPadding;
  if (outputPadding === undefined) {
    // for conv2d
    const outSize = Math.ceil(inputSize / stride);
    const neededInput = (outSize - 1) * stride + effectiveFilterSize;
    totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
  } else {
    // for convTranspose2d
    // totalPadding = beginning padding + ending padding
    // SAME_UPPER or SAME_LOWER mean pad the input so that
    //   output size = input size * strides
    // output size = (input size - 1) * stride + effectiveFilterSize
    //     - beginning padding - ending padding + output padding
    totalPadding = (inputSize - 1) * stride + effectiveFilterSize + outputPadding -
      inputSize * stride;
  }
  let paddingBegin;
  let paddingEnd;
  switch (autoPad) {
    case 'same-upper':
      paddingBegin = Math.floor(totalPadding / 2);
      paddingEnd = Math.floor((totalPadding + 1) / 2);
      break;
    case 'same-lower':
      paddingBegin = Math.floor((totalPadding + 1) / 2);
      paddingEnd = Math.floor(totalPadding / 2);
      break;
    default:
      throw new Error('The autoPad is invalid.');
  }
  return [paddingBegin, paddingEnd];
}
