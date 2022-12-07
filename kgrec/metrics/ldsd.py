import numpy as np
import progressbar as pb
import threading

from abc import ABC
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Sequence, Mapping, Iterable
from neo4j import Neo4jDriver, Result
from progressbar import ProgressBar

from kgrec.datasets import Dataset
from kgrec.metrics.metrics import SimilarityMetric, Pair, TSVWriter

_neighbourhood_query = '''
    OPTIONAL MATCH (x:$label)-[p]->(y:$label) 
    WHERE x.tsvID = $id
    WITH TYPE(p) as prop, collect(y) as do, count(distinct y) as cnt
    UNWIND do as neighbour
    RETURN neighbour.tsvID as neighbour, prop, 1 as type, cnt
    
    UNION
    
    OPTIONAL MATCH (y:$label)-[p]->(x:$label) 
    WHERE x.tsvID = $id
    WITH TYPE(p) as prop, collect(y) as di, count(distinct y) as cnt
    UNWIND di as neighbour
    RETURN neighbour.tsvID as neighbour, prop, 2 as type, cnt
    
    UNION
    
    OPTIONAL MATCH (x:$label)-[p]->(a:$label)<-[pv]-(y:$label)
    WHERE x.tsvID = $id and TYPE(p) = TYPE(pv)
    WITH TYPE(p) as prop, collect(y) as dio, count(distinct y) as cnt
    UNWIND dio as neighbour
    RETURN neighbour.tsvID as neighbour, prop, 3 as type, cnt
    
    
    UNION
    
    OPTIONAL MATCH (x:$label)<-[p]-(a:$label)-[pv]->(y:$label)
    WHERE x.tsvID = $id and TYPE(p) = TYPE(pv)
    WITH TYPE(p) as prop, collect(y) as dii, count(distinct y) as cnt
    UNWIND dii as neighbour
    RETURN neighbour.tsvID as neighbour, prop, 4 as type, cnt
    
    ORDER BY neighbour, prop, type
'''

Neighbour = namedtuple('Neighbour', 'id props')
Property = namedtuple('Property', 'id values')


class ResultIterator(Iterable, ABC):
    """ iterator over the results of Neo4J iterator """

    def __init__(self, r: Result):
        """
        creates a new result iterator to make it possible to iterate over all
        neighbours and their property links.

        :param r: the result over which an iterator shall be created.
        """
        self.r = iter(r)

    @staticmethod
    def _get_val(n: int) -> np.float:
        """
        computes the LDSD value for the number of links.

        :param n: the number of links.
        :return: the computed LDSD value for the number of links.
        """
        return np.float(1.0) / (np.float(1.0) + np.log(n))

    @staticmethod
    def _get_prop_tuple(prop_id: int, val_map: Mapping[int, int]) -> Property:
        """
        gets the property tuple for the specified information.

        :param prop_id: the unique ID of the property.
        :param val_map: a map of the different LDSD link counts.
        :return: the property tuple with the computed LDSD values.
        """
        values = {k: ResultIterator._get_val(v) for k, v in val_map.items()}
        return Property(prop_id, values)

    def __iter__(self):
        record = next(self.r, None)
        while record is not None:
            neighbour_id = record['neighbour']
            props = []
            while record is not None and record['neighbour'] == neighbour_id:
                property_id = record['prop']
                val_map = {}
                while record is not None and record['prop'] == property_id:
                    val_map[record['type']] = record['cnt']
                    t = ResultIterator._get_prop_tuple(prop_id=property_id,
                                                       val_map=val_map)
                    props.append(t)
                    record = next(self.r, None)
            yield Neighbour(neighbour_id, props)


class Counter:
    """ a counter to keep track of processed entities (thread-safe) """

    def __init__(self, progress_bar: ProgressBar):
        """
        creates a thread-safe wrapper for the progress bar.

        :param progress_bar: for which this counter is a thread-safe wrapper.
        """
        self._progress_bar = progress_bar
        self._lock = threading.Lock()
        self._n = 0

    def increment(self):
        """ indicates that another entity was processed """
        self._lock.acquire()
        try:
            self._n += 1
            self._progress_bar.update(self._n)
        finally:
            self._lock.release()


class LDSD(SimilarityMetric):
    """ this class implements Linked Data Semantic Distance """

    def __init__(self, dataset: Dataset, neo4j_driver: Neo4jDriver,
                 num_of_jobs: int = 1):
        super(LDSD, self).__init__('ldsd', dataset, neo4j_driver)
        self._num_of_jobs = num_of_jobs

    @staticmethod
    def _collect_neighbours(tx, x_id: int, label: str) -> Sequence[Neighbour]:
        r = tx.run(_neighbourhood_query.replace('$label', label), id=x_id)
        return [n for n in ResultIterator(r)]

    @staticmethod
    def _compute_metric_value(neighbour: Neighbour) -> np.float:
        link_values = np.zeros(4)
        for p in neighbour.props:
            for k, v in p.values.items():
                link_values[k - 1] += v
        return np.float(1.0) / (np.float(1.0) + np.sum(link_values))

    def _run_computation(self, x_ids: Sequence[int], c: Counter,
                         writer: TSVWriter):
        with self.driver.session() as session:
            for x_id in x_ids:
                neighbours = session.write_transaction(
                    self._collect_neighbours, x_id,
                    self.dataset.capitalized_name,
                )
                pairs = [Pair(x_id, neighbour.id,
                              self._compute_metric_value(neighbour))
                         for neighbour in neighbours]
                pairs.append(Pair(x_id, x_id, np.float(0.0)))
                writer.push_pairs(pairs)
                c.increment()

    def _compute_pairs(self, writer: TSVWriter):
        index_values = self.dataset.index.index
        with pb.ProgressBar(max_value=len(index_values)) as progress_bar:
            c = Counter(progress_bar=progress_bar)
            with ThreadPoolExecutor(max_workers=self._num_of_jobs) as pool:
                futures = [pool.submit(self._run_computation, chunk, c, writer)
                           for chunk in
                           np.array_split(index_values, self._num_of_jobs)]
                wait(futures, return_when=ALL_COMPLETED)
