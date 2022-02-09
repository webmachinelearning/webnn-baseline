/**
 * Compute the beginning and ending pad given input, filter and stride.
 * @param {String} autoPad
 * @param {Number} inputSize
 * @param {Number} effectiveFilterSize
 * @param {Number} stride
 * @return {Array} [paddingBegin, paddingEnd]
 */
export function computePaddingForAutoPad(autoPad, inputSize, effectiveFilterSize, stride) {
  const outSize = Math.ceil(inputSize / stride);
  const neededInput = (outSize - 1) * stride + effectiveFilterSize;
  const totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
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
