import numpy as np
import pandas as pd

# Move this somewhere else
ROWS_PER_BATCH = 5

pd.concat(
    (
        pd.DataFrame(
            ["A"] * ROWS_PER_BATCH + ["B"] * ROWS_PER_BATCH,
            columns=["Batch"]
        ),
        pd.DataFrame(
            np.concatenate(
                (
                    np.random.normal(0.0, 1.0, (ROWS_PER_BATCH, 3)),
                    np.random.normal(100.0, 2.0, (ROWS_PER_BATCH, 3))
                )
            )
        )
    ),
    axis=1
).reset_index().rename(columns={"index": "Sample"})
