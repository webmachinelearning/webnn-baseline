How to generate test-data file for WPT tests?

Step 1: Please prepare resources JSON file which includes those tests
to test each operator of WebNN API without specified inputs and outputs
data.

Step 2: Implement generate test-data scripts

Step 3: Execute command for generating test-data files for WPT tests

```shell
node gen-operator-with-single-input.js resources\softsign.json
```

then, you can find two generated folders named 'test-data' and
'test-data-wpt'. There're raw test data as being
./test-data/softsign-data.json,
and raw WPT test-data file as being ./test-data-wpt/softsign.json.


You can manually modify some test data in
./test-data/softsign-data.json,
then execute Step 3, to update ./test-data-wpt/softsign.json.