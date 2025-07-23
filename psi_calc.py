import pandas as pd
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level = logging.INFO)


def calc_psi(values, bnds, shrs, eps=1e-6):
    if (bnds is np.nan) or (shrs is np.nan):
        return np.nan
    
    assert len(bnds) + 1 == len(shrs), f'len(bnds) + 1 == len(shrs), len(bnds) = {len(bnds)}, len(shrs) = {len(shrs)}'

    share_train = np.array(shrs)
    share_test = pd.Series(np.searchsorted(bnds, values)).value_counts(normalize=True).sort_index(ascending=True)
    share_test = share_test.add(pd.Series(np.zeros(len(shrs))), fill_value = 0)
    psi = (share_train - share_test) @ np.log((share_train + eps) / (share_test + eps))
    return psi


def get_split(values, *, n_buckets=10):
    if len(values) == 0:
        return np.nan, np.nan
    
#     threshold = 0.05

    if values.nunique() >= 10:
        n_buckets_psi = n_buckets
        _, bounds_psi = pd.qcut(values, np.linspace(0, 1, n_buckets_psi + 1)[1:-1], duplicates='drop', retbins=True)

        if len(bounds_psi) == 2 and bounds_psi[0] == values.min() and bounds_psi[1] == values.max():
            bounds_psi = [(bounds_psi[0] + bounds_psi[1]) / 2]
            
        if bounds_psi[-1] == values.max():
            bounds_psi = bounds_psi[:-1]
    else:
        unique = sorted(values.unique())
        bounds_psi = [(unique[i] + unique[i + 1]) / 2 for i in range(len(unique) - 1)]

    ###################
    shares_psi = pd.Series(np.searchsorted(bounds_psi, values, side='left')).value_counts(normalize=True).sort_index(ascending=True)
    empty_idxs = list(set(np.arange(len(bounds_psi) + 1)) - set(shares_psi.index))

    bounds_psi = np.delete(bounds_psi, empty_idxs)
#     print(bounds_psi)
    ###################

    shares_psi = pd.Series(np.searchsorted(bounds_psi, values, side='left')).value_counts(normalize=True).sort_index(ascending=True)
    
    assert len(bounds_psi) + 1 == len(shares_psi), f'len(bounds_psi) + 1 != len(shares_psi)'
    
    
    
    if len(bounds_psi) + 1 == len(shares_psi):
#         print(f'Разбиение прошло успешно')
        logging.info('Разбиение на бакеты для PSI/CSI прошло успешно')
#         print(f'Разбиение {feature} прошло успешно')
#         print(f'Границы {feature}:')
#         display(np.array(bounds_csi))
#         print(f'Доли {feature}:')
#         display(np.array(shares_csi))
#         print(20*'=')

    else:
#         print(20*'=')
#         print(f'!!!!! Ошибка при разбиении !!!!!')
#         print(20*'=')
        logging.info('!!!!!!!!!!Разбиение на бакеты для PSI/CSI: ошибка!!!!!!!!!!')

    return bounds_psi, shares_psi