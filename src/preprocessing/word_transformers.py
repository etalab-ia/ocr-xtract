import multiprocessing as mp
import re
from functools import partial
from math import ceil
import unidecode
import difflib

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm


class ParallelWordTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_jobs, postprocess=None):
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
        self.postprocess = postprocess

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            if self.postprocess is not None:
                X = self.fit(X, **fit_params).transform(X)
                if self.postprocess == "kbins":
                    self.postprocesser = KBinsDiscretizer(n_bins=5, strategy='kmeans', encode='ordinal')
                    return self.postprocesser.fit_transform(X)
            else:
                return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

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
            nb_cpu = self.n_jobs
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
        if hasattr(self, 'postprocesser'):
            res = self.postprocesser.transform(res)

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
    def __init__(self, n_jobs, postprocess=None):
        super().__init__(n_jobs, postprocess)
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

    def __init__(self, n_jobs, postprocess=None):
        super().__init__(n_jobs, postprocess)
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
        list_months = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre',
                       'novembre', 'decembre']
        for j in range(it):
            x = array[j]
            if len(re.findall(r"\d", str(x))) > 0:  # check only if there are digits otherwise it takes too long
                res[j] = True  # dateparser.parse(str(x).lower()) is not None
            else:
                if len(difflib.get_close_matches(unidecode.unidecode(str(x).lower()), list_months)) > 0:
                    res[j] = True
                else:
                    res[j] = False
        return res