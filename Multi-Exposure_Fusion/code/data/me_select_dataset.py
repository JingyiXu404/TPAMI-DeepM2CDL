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

from data.me_dataset import DatasetMultiExposure



def select_dataset(opt_dataset: Dict[str, Any], phase: str) -> Union[DatasetMultiExposure, List[DatasetMultiExposure]]:
    if opt_dataset['type'] == 'multi_exposure':
        D = DatasetMultiExposure
    else:
        raise NotImplementedError

    if phase == 'train':
        dataset = D(opt_dataset)
        return dataset
    else:
        datasets: List[DatasetMultiExposure] = []
        paths_X = glob(os.path.join(opt_dataset['dataroot_HX'], '*'))
        paths_LX = glob(os.path.join(opt_dataset['dataroot_LX'], '*'))
        paths_Guide = glob(os.path.join(opt_dataset['dataroot_HY'], '*'))
        down_scales = opt_dataset['down_scale']
        opt_dataset_sub = deepcopy(opt_dataset)
        for i in range(len(paths_X)):
            for scale in down_scales:
                opt_dataset_sub['dataroot_HX'] = paths_X[i]
                opt_dataset_sub['dataroot_LX'] = paths_LX[i]
                opt_dataset_sub['dataroot_HY'] = paths_Guide[i]
                opt_dataset_sub['down_scale'] = scale
                datasets.append(D(opt_dataset_sub))
        return datasets
