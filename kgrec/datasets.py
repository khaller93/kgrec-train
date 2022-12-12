import csv
import gzip
from collections import namedtuple

import pandas as pd

from os import listdir
from os.path import join, dirname, exists
from typing import Sequence, Mapping, Tuple, Set


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


class IRIFilter:
    """ a class maintaining a filter for relevant IRIs """

    def apply(self, iri: str) -> bool:
        """
        applies this filter, and checks whether given IRI is relevant.

        :param iri: which shall be checked whether an IRI is relevant.
        :return: `True`, if given IRI is relevant, otherwise `False`.
        """
        raise NotImplementedError('must be implemented by subclass')


class AllIRIFilter(IRIFilter):

    def apply(self, iri: str) -> bool:
        return True


class SetIRIFilter(IRIFilter):

    def __init__(self, entities: Set[str]):
        self._entities = entities

    def apply(self, iri: str) -> bool:
        return iri in self._entities


EntityIndex = namedtuple('EntityIndex', 'forward backward')


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
        self._index_pd = None
        self._rev_filter = None
        self._statements = None

    def index(self, check_for_relevance: bool = False) -> EntityIndex:
        """
        gets the index (number -> IRI) and reverse index (IRI -> number) of the
        KG for this dataset.

        :param check_for_relevance: if only index of relevant entities shall be
        returned.
        :return: number -> IRI index and IRI -> number reverse index.
        """
        if self._index is None:
            self._index = ({}, {})
            with gzip.open(join(_get_dataset_dir(self.name),
                                'index.tsv.gz'), 'rt') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    key = int(row[0])
                    iri = row[1]
                    self._index[0][key] = iri
                    self._index[1][iri] = key
        if not check_for_relevance:
            return EntityIndex(self._index[0], self._index[1])
        else:
            fil = self._relevant_entities_filter
            return EntityIndex(
                {k: v for k, v in self._index[0].items() if fil.apply(v)},
                {k: v for k, v in self._index[1].items() if fil.apply(k)})

    def index_iterator(self, check_for_relevance: bool = False):
        """
        gets an iterator over the index (number -> IRI) of the KG for this
        dataset.

        :param check_for_relevance: if only index of relevant entities shall be
        returned.
        :return: an iterator over the index (number -> IRI).
        """
        with gzip.open(join(_get_dataset_dir(self.name),
                            'index.tsv.gz'), 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            rev_filter = self._relevant_entities_filter
            for row in reader:
                if not check_for_relevance or rev_filter.apply(row[1]):
                    yield int(row[0]), row[1]

    @property
    def _relevant_entities_filter(self) -> IRIFilter:
        """
        gets the relevant entities, which are of interest for fetching vectors
        in the latent space.

        :return: pandas dataframe of relevant entities.
        """
        if self._rev_filter is None:
            ent_f = join(_get_dataset_dir(self.name),
                         'relevant_entities.tsv.gz')
            if exists(ent_f):
                entities = set()
                with gzip.open(ent_f, 'rt') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        entities.add(row[0])
                self._rev_filter = SetIRIFilter(entities)
            else:
                self._rev_filter = AllIRIFilter()

        return self._rev_filter

    def write_result_index(self, index_file_path: str,
                           only_relevant: bool = True):
        """
        writes the index to the specified file only considering the relevant
        entities per default. If the index of all entities shall be written,
        then set `only_relevant` to `False`.

        :param index_file_path: file path to index file.
        :param only_relevant: `True` is set per default, and this indicates that
        only the index of relevant entities is written to file, otherwise set
        `False`.
        """
        index_it = self.index(check_for_relevance=only_relevant) if \
            self._index is not None else \
            self.index_iterator(check_for_relevance=only_relevant)
        with gzip.open(index_file_path, 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in index_it:
                writer.writerow(row)

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
        self._index_pd = None
        self._rev_filter = None
        self._statements = None
