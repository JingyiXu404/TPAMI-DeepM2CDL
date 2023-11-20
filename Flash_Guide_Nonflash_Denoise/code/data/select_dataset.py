"""
--------------------------------------------
select dataset
--------------------------------------------
Hongyi Zheng (github: https://github.com/natezhenghy)
--------------------------------------------
Kai Zhang (github: https://github.com/cszn)
--------------------------------------------
"""

import os
from copy import deepcopy
from glob import glob
from typing import Any, Dict, List, Union

from data.dataset_denoising import DatasetDenoising



def select_dataset(opt_dataset: Dict[str, Any], phase: str) -> Union[DatasetDenoising, List[DatasetDenoising]]:
    if opt_dataset['type'] == 'denoising':
        D = DatasetDenoising
    else:
        raise NotImplementedError

    if phase == 'train':
        dataset = D(opt_dataset)
        return dataset
    else:
        datasets: List[DatasetDenoising] = []
        paths_X = glob(os.path.join(opt_dataset['dataroot_HX'], '*'))
        paths_Guide = glob(os.path.join(opt_dataset['dataroot_HY'], '*'))
        sigmas = opt_dataset['sigma']
        print(sigmas)
        opt_dataset_sub = deepcopy(opt_dataset)
        for i in range(len(paths_X)):
            for sigma in sigmas:
                opt_dataset_sub['dataroot_HX'] = paths_X[i]
                opt_dataset_sub['dataroot_HY'] = paths_Guide[i]
                opt_dataset_sub['sigma'] = sigma
                datasets.append(D(opt_dataset_sub))
        # for path in paths_X:
        #     for sigma in sigmas:
        #         opt_dataset_sub['dataroot_HX'] = path
        #         opt_dataset_sub['sigma'] = sigma
        #         datasets.append(D(opt_dataset_sub))
        # for path in paths_Guide:
        #     for sigma in sigmas:
        #         opt_dataset_sub['dataroot_HY'] = path
        #         opt_dataset_sub['sigma'] = sigma
        #         datasets.append(D(opt_dataset_sub))
        # print(len(datasets))
        return datasets
