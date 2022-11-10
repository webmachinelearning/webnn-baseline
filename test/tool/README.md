# Introduction
These Node.js scripts `gen-<operation-name>.js` in this `tool` folder are used to gerenate / update test data which would be involved into test data JSON files for WebNN operations tests of web-platform-tests project.

# Command to generate or update test data
```shell
node gen-<operation-name>.js resources\<operation-name>.json
```

Example for generating test data for `concat` operation tests as following:
```shell
node gen-concat.js resources\concat.json
```

Then you could see similar below output on the console, and find new / updated test data file under `test-data` folder.
```
[ Done ] Saved test data onto \path\webnn-baseline\test\tool\test-data\concat-data.json .
```