module.exports = {
  root: true,
  ignorePatterns: ['.eslintrc.js'],
  env: { 'es6': true, 'browser': true, 'node': true, 'mocha': true },
  parserOptions: { ecmaVersion: 2020, sourceType: 'module'},
  globals: {
    'chai': 'readonly',
  },
  rules: {
    'semi': 'error',
    'no-multi-spaces': ['error', { 'exceptions': { 'ArrayExpression': true } }],
    'indent': 2,
    'require-jsdoc': 'off',
    'max-len': ['error', {'code': 100}],
  },
  extends: [
    'eslint:recommended',
    'google',
  ],
}
