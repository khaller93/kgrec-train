import time

import numpy as np
import progressbar as pb
import threading

from abc import ABC
from collections import namedtuple
from typing import Sequence, Mapping, Iterable
from neo4j import Result, GraphDatabase, Neo4jDriver
from multiprocessing import Pool, Queue, Value

from kgrec.datasets import Dataset
from kgrec.metrics.graphdb import Neo4JDetails
from kgrec.metrics.metrics import SimilarityMetric, Pair
from kgrec.utils.widgets import widgets_with_label

_neighbourhood_query = '''
OPTIONAL MATCH (x:Resource)-[p]->(y:Resource) 
WHERE x.tsvID = $in_id
WITH TYPE(p) as prop, collect(y) as do, count(distinct y) as cnt
UNWIND do as neighbour
RETURN neighbour.tsvID as neighbour, neighbour.rvKey as key, prop, 1 as type,
cnt
    
UNION
    
OPTIONAL MATCH (y:Resource)-[p]->(x:Resource) 
WHERE x.tsvID = $in_id
WITH TYPE(p) as prop, collect(y) as di, count(distinct y) as cnt
UNWIND di as neighbour
RETURN neighbour.tsvID as neighbour, neighbour.rvKey as key, prop, 2 as type,
cnt

UNION
    
OPTIONAL MATCH (x:Resource)-[p]->(a:Resource)<-[pv]-(y:Resource)
WHERE x.tsvID = $in_id and TYPE(p) = TYPE(pv)
WITH TYPE(p) as prop, collect(y) as dio, count(distinct y) as cnt
UNWIND dio as neighbour
RETURN neighbour.tsvID as neighbour, neighbour.rvKey as key, prop, 3 as type,
cnt
        
UNION
    
OPTIONAL MATCH (x:Resource)<-[p]-(a:Resource)-[pv]->(y:Resource)
WHERE x.tsvID = $in_id and TYPE(p) = TYPE(pv)
WITH TYPE(p) as prop, collect(y) as dii, count(distinct y) as cnt
UNWIND dii as neighbour
RETURN neighbour.tsvID as neighbour, neighbour.rvKey as key, prop, 4 as type,
cnt
    
ORDER BY neighbour, prop, type
'''

_query = '''
CALL apoc.cypher.run("%s", {in_id: $id}) YIELD value
RETURN value.neighbour as neighbour, value.key as key, value.prop as prop,
value.type as type, value.cnt as cnt
ORDER BY neighbour, prop, type
''' % _neighbourhood_query.replace('\n', ' ')

Neighbour = namedtuple('Neighbour', 'id key props')
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
            neighbour_key = record['key']
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
            yield Neighbour(neighbour_id, neighbour_key, props)


class CounterListener:
    """ a counter to keep track of processed entities """

    def __init__(self, counter: Value, progress_bar: pb.ProgressBar):
        """
        creates a listener for the counter used by multiple threads.

        :param counter: a counter value that shall be observed.
        :param progress_bar: for which this counter is a thread-safe wrapper.
        """
        self._counter = counter
        self._progress_bar = progress_bar
        self._t = None
        self._closed = False

    def __enter__(self):
        self._t = threading.Thread(target=self._observe)
        self._t.start()
        return self

    def _observe(self):
        """ observes the counter which is shared among multiple processes and
         updates the progress bar accordingly """
        while not self._closed:
            with self._counter.get_lock():
                n = self._counter.value
                if n > 0:
                    self._progress_bar.update(n)
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        self._t.join()


class _Payload:
    """ a payload for computing LDSD metric """

    def __init__(self, dataset_name: str, neo4j_details: Neo4JDetails):
        self.dataset_name = dataset_name
        self.neo4j_details = neo4j_details
        self.chunk = None

    @property
    def driver(self) -> Neo4jDriver:
        return GraphDatabase.driver(self.neo4j_details.bolt_url,
                                    auth=self.neo4j_details.auth)

    def with_chunk(self, chunk: Sequence[int]):
        obj = _Payload(self.dataset_name, self.neo4j_details)
        obj.chunk = chunk
        return obj


class LDSD(SimilarityMetric):
    """ this class implements Linked Data Semantic Distance """

    def __init__(self, neo4j_details: Neo4JDetails, dataset: Dataset,
                 num_of_jobs: int = 1):
        super(LDSD, self).__init__('ldsd', dataset)
        self._neo4j_details = neo4j_details
        self._num_of_jobs = num_of_jobs

    @staticmethod
    def _collect_neighbours(tx, x_id: int) -> Sequence[Neighbour]:
        r = tx.run(_query, id=x_id)
        return [n for n in ResultIterator(r)]

    @staticmethod
    def _compute_metric_value(neighbour: Neighbour) -> np.float:
        link_values = np.zeros(4)
        for p in neighbour.props:
            for k, v in p.values.items():
                link_values[k - 1] += v
        return np.float(1.0) / (np.float(1.0) + np.sum(link_values))

    @staticmethod
    def _export_global_vars(queue: Queue, counter: Value):
        global _pair_queue
        global _counter
        _pair_queue = queue
        _counter = counter

    @staticmethod
    def _run_computation(payload: _Payload):
        with payload.driver.session() as session:
            for x_id, key in payload.chunk:
                neighbours = session.write_transaction(
                    LDSD._collect_neighbours, int(x_id),
                )
                pairs = [Pair(key, neighbour.key,
                              LDSD._compute_metric_value(neighbour))
                         for neighbour in neighbours if neighbour.key != -1]
                pairs.append(Pair(key, key, np.float(0.0)))
                _pair_queue.put(pairs)
                with _counter.get_lock():
                    _counter.value += 1

    def _compute_pairs(self, queue: Queue):
        values = [(row['key_index'], int(i)) for i, row in
                  self.dataset.relevant_entities.iterrows()]
        ds_name = self.dataset.capitalized_name
        with pb.ProgressBar(max_value=len(values),
                            widgets=widgets_with_label('Compute LDSD:')
                            ) as progress_bar:
            counter = Value('i', 0)
            with CounterListener(counter=counter, progress_bar=progress_bar):
                with Pool(processes=self._num_of_jobs,
                          initializer=LDSD._export_global_vars,
                          initargs=(queue, counter,)) as p:
                    payload = _Payload(ds_name, self._neo4j_details)
                    params = [payload.with_chunk(chunk) for chunk in
                              np.array_split(values, self._num_of_jobs)]
                    p.map(self._run_computation, params)
