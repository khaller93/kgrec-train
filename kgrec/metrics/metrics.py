from collections import namedtuple
from gzip import GzipFile
from os import makedirs
from queue import Queue, Empty
from threading import Thread
from typing import Sequence

from kgrec.datasets import Dataset
from neo4j import Neo4jDriver
from os.path import join, exists

Pair = namedtuple('Pair', 'a b val')


class TSVWriter:
    """ a thread-safe writer for similarity pairs into TSV files """

    def __init__(self, name: str, dataset: Dataset, out_dir_path: str):
        """
        creates a new writer which writes values to a TSV file.

        :param name: the name of the metric for which values shall be written.
        :param dataset: for which the values are computed.
        :param out_dir_path: to which the values shall be written.
        """
        self._name = name
        self._dataset = dataset
        self._out_dir_path = out_dir_path
        self._queue = Queue(maxsize=0)
        self._closed = False
        self._t = None

    def __enter__(self):
        self._t = Thread(target=self._run_write)
        self._t.start()
        return self

    def push_pairs(self, pairs: Sequence[Pair]):
        """
        pushes the given list of pairs to be written to the TSV file.

        :param pairs: that shall be written to the TSV file.
        """
        for p in pairs:
            self._queue.put(p, block=False)

    def _run_write(self):
        """ a method writing pairs of the queue to TSV file """
        if not exists(self._out_dir_path):
            makedirs(self._out_dir_path)
        f_path = join(self._out_dir_path, '%s.tsv.gz' % self._name)
        with GzipFile(filename=f_path, mode='w') as gf:
            while not self._closed or self._queue.qsize() != 0:
                try:
                    item = self._queue.get(block=True, timeout=0.1)
                    gf.write(bytes('%d\t%d\t%f\n' % (item.a, item.b, item.val),
                                   'utf-8'))
                except Empty:
                    continue

    def _write_index(self):
        """ writes the entity index of dataset, if its not already existing """
        index_f = join(self._out_dir_path, 'entities.tsv.gz')
        if not exists(index_f):
            df = self._dataset.relevant_entities.copy()
            del df['key_index']
            df.to_csv(index_f, header=False, sep='\t', compression='gzip')

    def __exit__(self, exception_type, exception_value, traceback):
        self._write_index()
        self._closed = True
        self._t.join()


class SimilarityMetric:
    """ a similarity metric using graph analytics """

    def __init__(self, name: str, dataset: Dataset, neo4j_driver: Neo4jDriver):
        """
        creates a similarity metric with the specified name and for the given
        dataset. The passed Neo4J driver will be used to create sessions for
        querying the graph DB.

        :param name: unique name of the metric.
        :param dataset: for which the similarity metric shall be computed.
        :param neo4j_driver: driver to Neo4J.
        """
        self.name = name
        self.dataset = dataset
        self.driver = neo4j_driver

    def _compute_pairs(self, writer: TSVWriter):
        """
        computes the similarity value for pairs and uses the specified writer
        to write those values to a TSV file.

        :param writer: to which the pairs shall be written.
        """
        raise NotImplementedError('must be implemented by the subclass')

    def compute(self, model_dir_path: str):
        """
        starts the computation of the similarity metric for pairs and writes the
        results to the specified directory.

        :param model_dir_path: the directory to write to.
        """
        with TSVWriter(self.name, self.dataset,
                       join(model_dir_path, self.dataset.name)) as writer:
            self._compute_pairs(writer)
