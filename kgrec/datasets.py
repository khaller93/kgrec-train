import csv
import gzip

import numpy as np
import pandas as pd

from os import listdir
from os.path import join, dirname, exists
from typing import Sequence


def _get_data_dir() -> str:
    """
    gets the path to the data directory with the datasets.

    :return: the path of the data directory with the datasets.
    """
    return join(dirname(dirname(__file__)), 'data')


def _get_dataset_dir(name: str) -> str:
    """
    gets the path to the dataset directory with the kg.

    :param name: name of the dataset.
    :return: the path of the dataset directory with the kg.
    """
    return join(_get_data_dir(), name)


class Dataset:
    """ a dataset representing a KG to train recommendations on """

    @staticmethod
    def supported_datasets() -> Sequence[str]:
        """
        gets the names of supported datasets.

        :return: a sequence of names of supported datasets.
        """
        return listdir(_get_data_dir())

    @staticmethod
    def get_dataset_for(name: str):
        """
        gets the dataset known under the specified name. If no such dataset is
        known, then None will be returned.

        :param name: name of dataset that shall be returned.
        :return: dataset known under this name, or None, if no supported dataset
        isn't specified under this name.
        """
        name = name.lower()
        if name not in Dataset.supported_datasets():
            return None
        return Dataset(name)

    def __init__(self, name: str):
        """
        creates a new dataset with the given unique name.

        :param name: unique name of this dataset.
        """
        self.name = name
        self._index = None
        self._relevant_entities = None
        self._statements = None

    @property
    def capitalized_name(self):
        """
        gets the capitalized name of this dataset.

        :return: capitalized name of this dataset.
        """
        return self.name.capitalize()

    @property
    def index(self) -> pd.DataFrame:
        """
        gets the index (number -> IRI) of the KG for this dataset.

        :return: pandas dataframe of the number -> IRI index.
        """
        if self._index is None:
            self._index = pd.read_csv(join(_get_dataset_dir(self.name),
                                           'index.tsv.gz'),
                                      names=['index', 'iri'],
                                      sep='\t', header=None,
                                      compression='gzip').set_index('index')
        return self._index

    def get_index_for(self, iri: str) -> int:
        """
        gets the index number for the specified IRI. -1 will be returned, if
        the specified iri can't be found.

        :param iri: for which the index shall be fetched.
        :return: index of the given iri, or -1, if this iri can't be found.
        """
        index_list = self.index.index[self.index.index['iri'] == iri].tolist()
        if len(index_list) == 0:
            return -1
        return index_list[0]

    @property
    def relevant_entities(self) -> pd.DataFrame:
        """
        gets the relevant entities, which are of interest for fetching vectors
        in the latent space.

        :return: pandas dataframe of relevant entities.
        """
        if self._relevant_entities is None:
            ent_f = join(_get_dataset_dir(self.name), 'relevant_entities.tsv.gz')
            if exists(ent_f):
                df = pd.read_csv(ent_f, names=['iri'], sep='\t', header=None,
                                 compression='gzip')
                idf = self.index.copy()
                idf['key_index'] = idf.index
                df['key_index'] = df[['iri']] \
                    .merge(idf, on='iri', how='left')['key_index']
                df = df[df['key_index'].notnull()]
                df['key_index'] = df['key_index'].astype(np.int)
                df = df.reset_index(drop=True)
                self._relevant_entities = df
            else:
                df = self.index.copy()
                df['key_index'] = df.index
                self._relevant_entities = df
        return self._relevant_entities

    @property
    def statements(self) -> pd.DataFrame:
        """
        gets the statements of the KG for this dataset.

        :return: pandas dataframe of statements of the KG for this dataset.
        """
        if self._statements is None:
            self._statements = pd.read_csv(join(_get_dataset_dir(self.name),
                                                'statements.tsv.gz'),
                                           names=['subj', 'pred', 'obj'],
                                           sep='\t', header=None,
                                           compression='gzip')
        return self._statements

    def statement_iterator(self):
        """
        gets an iterator over the statements of this dataset.

        :return: an iterator over the statements of this dataset.
        """
        with gzip.open(join(_get_dataset_dir(self.name),
                            'statements.tsv.gz'), 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                yield [int(x) for x in row]

    def free(self):
        """ frees the resources loaded for this dataset """
        self._index = None
        self._statements = None
