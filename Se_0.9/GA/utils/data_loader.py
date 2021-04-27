#  nhan vao input la mot file
# xu ly ssa
# tra ve dang du lieu mong muon
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from utils.ssa import SSA
from utils.reprocess_daily import extract_data, ed_extract_data, roll_data


def get_input_data(input_file, default_n, sigma_lst):
    dat = pd.read_csv(input_file, header=0)
    Q = dat['Q'].to_list()
    H = dat['H'].to_list()

    # print(Q[:5])
    # print(H[:5])
    lst_H_ssa = SSA(H, default_n)
    lst_Q_ssa = SSA(Q, default_n)

    H_ssa = lst_H_ssa.reconstruct(sigma_lst)
    Q_ssa = lst_Q_ssa.reconstruct(sigma_lst)
    # print(Q_ssa[:5])
    # print(H_ssa[:5])
    # dat['Q_ssa'] = Q_ssa
    # dat['H_ssa'] = H_ssa

    dat['Q_ssa'] = Q_ssa
    # print(dat['Q'][:5])
    dat['H_ssa'] = H_ssa
    # print(dat['H'][:5])

    # print(dat.head())
    result = dat[['Q', 'H', 'Q_ssa', 'H_ssa']]
    return result


if __name__ == "__main__":
    res = get_input_data('../data/SonTay.csv', 20, [1, 2, 3])
    res.to_csv('../data/modified_data.csv', index=False)
    print(res.head())
