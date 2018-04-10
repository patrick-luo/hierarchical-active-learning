# hierarchical-active-learning
Hierarchical active learning with proportion feedback on regions

This is initially for the submission for ECML 2018.

run_one_fold.py is the main program;

classes.py defines some useful objects;

util.py defines some handful tools.

data/ is a directory for sample dataset 'Wine'

To run the sample data for 200 queries.

run command: python2.7 data/ 200 

After that, there will be three more files generated:

'acc' records the accurary for the past 200 queries;

'auc' records the auc scores for the past 200 queries;

'frr' records the feature reduction rate for the past 200 queries;
