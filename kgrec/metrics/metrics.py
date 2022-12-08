from collections import namedtuple
from gzip import GzipFile
from multiprocessing import Queue
from os import makedirs
from queue import Empty
from threading import Thread
from typing import Sequence

from kgrec.datasets import Dataset
from os.path import join, exists

Pair = namedtuple('Pair', 'a b val')


class TSVWriter:
    """ a thread-safe writer for similarity pairs into TSV files """

    def __init__(self, name: str, out_dir_path: str, queue: Queue):
        """
        creates a new writer which writes values to a TSV file.

        :param name: the name of the metric for which values shall be written.
        :param out_dir_path: to which the values shall be written.
        :param queue: queue from which the pairs to write shall be read.
        """
        self._name = name
        self._out_dir_path = out_dir_path
        self._queue = queue
        self._t = None
        self._closed = False
        self._first = True

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

    def _write_item(self, file, item):
        if self._first:
            file.write(bytes('%d\t%d\t%f' % (item.a, item.b, item.val),
                             'utf-8'))
            self._first = False
        else:
            file.write(bytes('\n%d\t%d\t%f' % (item.a, item.b, item.val),
                             'utf-8'))

    def _run_write(self):
        """ a method writing pairs of the queue to TSV file """
        if not exists(self._out_dir_path):
            makedirs(self._out_dir_path)
        f_path = join(self._out_dir_path, '%s.tsv.gz' % self._name)
        with GzipFile(filename=f_path, mode='w') as gf:
            while not self._closed or not self._queue.empty():
                try:
                    obj = self._queue.get(block=True, timeout=1)
                    if isinstance(obj, list):
                        for item in obj:
                            self._write_item(gf, item)
                    else:
                        self._write_item(gf, obj)
                except Empty:
                    continue

    def __exit__(self, exception_type, exception_value, traceback):
        self._closed = True
        self._t.join()


class SimilarityMetric:
    """ a similarity metric using graph analytics """

    def __init__(self, name: str, dataset: Dataset):
        """
        creates a similarity metric with the specified name and for the given
        dataset. The passed Neo4J driver will be used to create sessions for
        querying the graph DB.

        :param name: unique name of the metric.
        :param dataset: for which the similarity metric shall be computed.
        """
        self.name = name
        self.dataset = dataset

    def _compute_pairs(self, queue: Queue):
        """
        computes the similarity value for pairs.

        :param queue: to which the computed pairs shall be pushed.
        """
        raise NotImplementedError('must be implemented by the subclass')

    def _write_index(self, out_dir_path: str):
        """ writes the entity index of dataset, if its not already existing """
        index_f = join(out_dir_path, 'entities.tsv.gz')
        if not exists(index_f):
            df = self.dataset.relevant_entities.copy()
            del df['key_index']
            df.to_csv(index_f, header=False, sep='\t', compression='gzip')

    def compute(self, model_dir_path: str):
        """
        starts the computation of the similarity metric for pairs and writes the
        results to the specified directory.

        :param model_dir_path: the directory to write to.
        """
        queue = Queue(maxsize=0)
        out_dir_path = join(model_dir_path, self.dataset.name)
        with TSVWriter(self.name, out_dir_path, queue):
            self._compute_pairs(queue)
        self._write_index(out_dir_path)
