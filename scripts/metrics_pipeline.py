"""
TODO: make a big, nice pipeline that takes in one input file & runs it
through Confounded, ComBat, and SVA, saves those output files, and
calculates the following metrics:
- how long did it take?
- for each algorithm, what's the MSE (before - after)?
- what is our Random Forests & SVA accuracy for batch before & after?
- what is our "" accuracy for true signal before & after?
"""

INPUT_FILE = "../data/GSE40292_copy.csv"
BATCH_COL = "Batch"

# TODO: Run through Confounded & Save output file

# TODO: Classify on batch, record accuracies & time

# TODO: Classify on true signal, record accuracies & time

# TODO: save / output metrics
