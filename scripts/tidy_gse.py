import pandas as pd

gseid = "GSE37199"
path = "data/avery/{}/".format(gseid)

# clinical = pd.read_table("{}{}_Clinical.txt".format(path, gseid))
unadj = pd.read_table("{}{}_Expression_BatchUnadjusted.txt.gz".format(path, gseid)).rename(columns={"Unnamed: 0": "Sample"})
clinical = pd.read_table("{}{}_Clinical.txt".format(path, gseid), index_col=False)

tot = pd.concat([clinical.set_index("SampleID"), unadj.set_index("Sample")], axis="columns", join="inner").reset_index().rename(columns={"index": "Sample"})

tot.to_csv("{}clinical.csv".format(path), index=False)
