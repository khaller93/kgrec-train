import csv
import gzip
import logging
import re
from collections import namedtuple

import pandas as pd

from os import listdir, makedirs
from os.path import join, dirname, exists
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory
from typing import Sequence, Set


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


class _RelevantEntityCollector:
    """ a class to collect all relevant entities into a set and write the set
     to a file """

    def __init__(self, relevant_entity_file_path: str):
        """

        :param relevant_entity_file_path:
        """
        self._relevant_entity_file_path = relevant_entity_file_path
        self._entity_set = set()

    def __enter__(self):
        self._f = gzip.open(self._relevant_entity_file_path, 'wt')
        self._writer = csv.writer(self._f, delimiter='\t')
        return self

    def push(self, entity_name: str):
        """
        pushes the entity with the given name to the set of relevant entities.
        If the same entity has been pushed before, then this method is doing
        nothing.

        :param entity_name: name of the entity which shall be pushed to the
        set of relevant entities.
        """
        if entity_name not in self._entity_set:
            self._writer.writerow([entity_name])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.__exit__(exc_type, exc_val, exc_tb)
        self._entity_set = None


class _EntityIDMapper:
    """ a class to map entities to IDs and write the mapping to a file """

    def __init__(self, index_file_path: str):
        """
        creates a new entity ID mapper for mapping entity to ID numbers, and
        writing the mapping to the specified file path.

        :param index_file_path: path to the file to which the mapping shall be
        written.
        """
        self._index_file_path = index_file_path
        self._n = 0
        self._entity_mapping = {}

    def __enter__(self):
        self._f = gzip.open(self._index_file_path, 'wt')
        self._writer = csv.writer(self._f, delimiter='\t')
        return self

    def get_id(self, entity_name: str) -> int:
        """
        gets the ID number for the entity with the given name.

        :param entity_name: entity name for which to get the ID number.
        :return: the ID number for the given entity name.
        """
        if entity_name not in self._entity_mapping:
            self._writer.writerow([self._n, entity_name])
            self._entity_mapping[entity_name] = self._n
            self._n += 1
        return self._entity_mapping[entity_name]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.__exit__(exc_type, exc_val, exc_tb)
        self._entity_mapping = None


class PyKeenDatasetFetcher:
    """ a class that fetches a PyKeen dataset """

    def __init__(self, dataset_name: str):
        """
        creates a new fetcher for the PyKeen dataset with the given name.

        :param dataset_name: name of the PyKeen dataset.
        """
        self._dataset_name = dataset_name

    def _write_dataset(self, triples_factories: Sequence[TriplesFactory],
                       index_file_path: str,
                       rev_entities_file_path: str,
                       statements_file_path: str):
        """
        writes triples factories of the dataset to disk in proper format.

        :param triples_factories: list of triples factories.
        :param index_file_path: file path to which the index shall be written.
        :param rev_entities_file_path: file path to which the relevant entities
        shall be written.
        :param statements_file_path: file path to which the statements shall be
        written.
        """
        n = 0
        with _EntityIDMapper(index_file_path) as mapper:
            with _RelevantEntityCollector(rev_entities_file_path) as collector:
                with gzip.open(statements_file_path, 'wt') as stmt_writer:
                    w = csv.writer(stmt_writer, delimiter='\t')
                    for tf in triples_factories:
                        for triple in tf.label_triples(tf.mapped_triples):
                            w.writerow([mapper.get_id(triple[0]),
                                        mapper.get_id(triple[1]),
                                        mapper.get_id(triple[2])])
                            collector.push(triple[0])
                            collector.push(triple[2])
                            n += 1
        logging.info('Wrote %d statements of PyKeen dataset "%s"'
                     % (n, self._dataset_name))

    def run(self):
        """ fetches data for the specified dataset and writes it to disk in the
        proper format """
        pykeen_dataset = get_dataset(dataset=self._dataset_name)
        if pykeen_dataset is not None:
            dataset_wd = _get_dataset_dir(self.dataset_id)
            index_file = join(dataset_wd, 'index.tsv.gz')
            statements_file = join(dataset_wd, 'statements.tsv.gz')
            rev_entities_file = join(dataset_wd, 'relevant_entities.tsv.gz')
            if not (exists(index_file) and exists(statements_file)
                    and exists(rev_entities_file)):
                logging.info('PyKeen dataset "%s" must be fetched'
                             % self._dataset_name)
                if not exists(dataset_wd):
                    makedirs(dataset_wd)
                tf_list = [pykeen_dataset.training, pykeen_dataset.testing,
                           pykeen_dataset.validation]
                # check if sets are triples factories.
                for tf in tf_list:
                    if not isinstance(tf, TriplesFactory):
                        raise ValueError(
                            'format of PyKeen dataset "%s" isn\'t supported'
                            % self._dataset_name)
                # write to data folder
                self._write_dataset(tf_list, index_file, rev_entities_file,
                                    statements_file)
            else:
                logging.debug('PyKeen dataset "%s" has already been fetched'
                              % self._dataset_name)
        else:
            logging.info('PyKeen dataset "%s" is unknown' % self._dataset_name)

    @property
    def dataset_id(self):
        """
        gets the unique name of this Pykeen dataset as ID.

        :return: unique name of this dataset.
        """
        return 'pykeen_%s' % re.sub(r'\W', '_', self._dataset_name).lower()


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
        if name.startswith('pykeen:'):
            fetcher = PyKeenDatasetFetcher(name.replace('pykeen:', ''))
            fetcher.run()
            return Dataset(fetcher.dataset_id)
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
