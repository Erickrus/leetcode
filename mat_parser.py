# -*- coding: utf-8 -*-
import json
import scipy.io
import numpy as np

class MatParser:
    def trans(self, item):
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, dict):
            res = {}
            for key in item.keys():
                res[key] = self.trans(item[key])
        elif isinstance(item, list):
            res = []
            for i in range(len(item)):
                res.append(self.trans(item[i]))
        elif isinstance(item, bytes):
            res = item.decode(encoding="utf-8", errors="ignore")
        else:
            res = item
        return res

    def parse(self, filename):
        data = scipy.io.loadmat(filename)
        data = self.trans(data)
        return data

if __name__ == "__main__":
    mp = MatParser()
    m = mp.parse("./300W_LP/AFW/AFW_2060241469_1_2.mat")
    print(json.dumps(m, indent=2))
