import pandas as pd

gseid = "GSE40292"
path = "data/avery/{}/".format(gseid)

# clinical = pd.read_table("{}{}_Clinical.txt".format(path, gseid))
unadj = pd.read_table("{}{}_Expression_BatchUnadjusted.txt.gz".format(path, gseid)).rename(columns={"Unnamed: 0": "Sample"})
batch = pd.read_table("{}{}_Expression_Batch.txt".format(path, gseid)).drop("sample.1", axis="columns")
batch = batch.dropna(subset=['Batch'])
try:
    batch["Batch"] = batch['Batch'].astype(int)
except:
    pass
tot = pd.concat([batch.set_index("sample"), unadj.set_index("Sample")], axis="columns", join="inner").reset_index().rename(columns={"index": "Sample"})

tot.to_csv("{}tidy.csv".format(path), index=False)
