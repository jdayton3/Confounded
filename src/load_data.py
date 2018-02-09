import numpy as np
import gzip

class RNASeq(object):
    def __init__(self):
        self.data = self.load_data()
        self.cur_ix = 0

    def load_data(self):
        data = []
        with gzip.open("./data/RNASeq.txt.gz", "r") as infile:
            header = infile.readline()
            for line in infile:
                line = line.strip("\n").split("\t")[1:]
                line = [float(x) for x in line]
                data.append(line)
        return np.array(data)

    def next_batch(self, batch_size):
        if batch_size > len(self.data):
            raise ValueError("Argument `batch_size` greater than length of dataset.")
        if self.cur_ix + batch_size > len(self.data):
            np.random.shuffle(self.data)
            self.cur_ix = 0
        batch = self.data[self.cur_ix:self.cur_ix + batch_size]
        self.cur_ix += batch_size
        return batch

if __name__ == "__main__":
    data = RNASeq()
    print(data.data.shape)