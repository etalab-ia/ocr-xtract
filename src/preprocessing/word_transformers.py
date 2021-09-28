import multiprocessing as mp
import re
from functools import partial
from math import ceil

import dateparser
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm


class ParallelWordTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def func(self, res: np.array, it: int, array):
        return array

    def fmp(self, i, nb_blocks, x):
        it = ceil(len(x) / nb_blocks)
        test_data = x[i*it:(i+1)*it]
        res = np.zeros(len(test_data))
        self.func(res, len(test_data), test_data)
        return res

    def get_feature_names(self):
        return [self.__class__.__name__]

    def transform(self, X: pd.DataFrame):
        # pandarallel.initialize(progress_bar=True)
        print(f"Transforming with {self.__class__.__name__}")
        if len(X['document_name'].unique()) > 1: #this is for training or testing
            nb_cpu = mp.cpu_count()
        else: #this is for the inference
            nb_cpu = 1 #limit the number of cpu to 1 to avoid memory issues in production
        print(f'Number of CPU used by {self.__class__.__name__}: {nb_cpu}')

        array = X['word'].to_numpy()
        nb_blocks = nb_cpu * 2
        with mp.Pool(nb_cpu) as pool:
            arrays = list(
                tqdm(pool.imap(partial(self.fmp, nb_blocks=nb_blocks, x=array), range(nb_blocks)), total=nb_blocks))
        res = np.hstack(arrays)
        res.resize((len(res), 1))
        print('Done ! ')
        return res


class ContainsDigit(ParallelWordTransformer):
    """
    Check if a string contains digits
    """
    def func(self, res, it, array):
        for j in range(it):
            x = array[j]
            res[j] = len(re.findall(r"\d", str(x))) > 0


class IsPrenom(ParallelWordTransformer):
    """
    Check if a string is a prenom. The list of prenom is the french prenom from INSEE dataset
    """
    # TODO : is Nom Is Prenom should
    def __init__(self):
        super().__init__()
        with open('src/preprocessing/prenoms.txt', 'r', encoding='UTf_8') as f:
            self.prenom_dict = {line.lower().split(',')[0]: int(line.lower().split(',')[1].strip()) for line in
                           f.readlines()}

    def func(self, res, it, array):
        for j in range(it):
            x = array[j]
            try:
                res[j] = self.prenom_dict[str(x).lower()]
            except:
                res[j] = False


class IsNom (ParallelWordTransformer):
    """
    Check if a string is a nom. The list of nom is the french nom from INSEE dataset
    """

    def __init__(self):
        super().__init__()
        with open('src/preprocessing/noms.txt', 'r') as f:
            self.nom_dict = {line.lower().split(',')[0]: int(line.lower().split(',')[1].strip()) for line in
                           f.readlines()}


    def func(self,  res, it, array):
        for j in range(it):
            x = array[j]
            try:
                res[j] = self.nom_dict[str(x).lower()]
            except:
                res[j] = False


class IsDate (ParallelWordTransformer):
    def func(self, res, it, array):
        for j in range(it):
            x = array[j]
            if len(re.findall(r"\d", str(x))) > 0:  # check  only if there are digits otherwise it takes too long
                res[j] = dateparser.parse(str(x).lower()) is not None
            else:
                res[j] = False
        return res