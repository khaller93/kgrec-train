import math
from typing import Mapping, Sequence

import numpy as np
import progressbar as pb

from neo4j import Neo4jDriver, Record, Session
from kgrec.datasets import Dataset


class Initializer:
    """ Initializer for Neo4J, which loads the statements into the database """

    def __init__(self, dataset: Dataset, driver: Neo4jDriver,
                 batch_size: int = 100000):
        self._dataset = dataset
        self._driver = driver
        self._batch_size = batch_size

    def _create_entities(self, session: Session) -> Mapping[int, int]:
        """ loads entities of the dataset into the Neo4J graph database """
        ds_name = self._dataset.capitalized_name
        print('Load entities:')
        nodes = {}
        size = len(self._dataset.index)
        with pb.ProgressBar(max_value=size) as p:
            n = 0
            p.update(n)
            chunks = np.array_split(self._dataset.index,
                                    math.ceil(float(size) / self._batch_size))
            for chunk in chunks:
                n_d = session.write_transaction(self._write_nodes, ds_name,
                                                [e for e in chunk.index])
                for record in n_d:
                    nodes[record[1]] = record[0]

                n += len(chunk)
                p.update(n)

            return nodes

    def _create_statement(self, node_dict: Mapping[int, int], session: Session):
        """ loads statements of the dataset into the Neo4J graph database """
        ds_name = self._dataset.capitalized_name
        print('Load statements:')
        size = len(self._dataset.statements)
        with pb.ProgressBar(max_value=size) as p:
            n = 0
            p.update(n)
            chunks = np.array_split(self._dataset.statements,
                                    math.ceil(float(size) / self._batch_size))
            for chunk in chunks:
                statements = ['[%d,\'p%d\',%d]' % (node_dict[int(row['subj'])],
                                                   int(row['pred']),
                                                   node_dict[int(row['obj'])])
                              for i, row in chunk.iterrows()]
                session.write_transaction(self._write_edges, ds_name,
                                          statements)

    def load(self):
        """ loads the given statements into the database with the specified name
        """
        with self._driver.session() as session:
            entity_map = self._create_entities(session)
            self._create_statement(entity_map, session)

    def clear(self):
        """ deletes all the nodes and statements loaded into the Neo4J database
        """
        with self._driver.session() as session:
            session.write_transaction(self._delete_all,
                                      self._dataset.capitalized_name)

    @staticmethod
    def _delete_all(tx, label: str):
        r = tx.run('''
        MATCH (n:%s)
        CALL {
          WITH n
          DETACH DELETE n
        }
        ''' % label)
        return r.single()

    @staticmethod
    def _write_nodes(tx, label: str, chunk: Sequence[int]) -> Sequence[Record]:
        result = tx.run(('''
        UNWIND [%s] as node_id
        CREATE (n:Resource:$label)
        SET n.tsv_id = node_id
        RETURN id(n) as node_id, n.tsv_id as tsv_id
        ''' % ','.join(map(str, chunk))).replace('$label', label))
        return result.values()

    @staticmethod
    def _write_edges(tx, label: str, statements: Sequence[str]):
        tx.run(('''
        UNWIND[%s] as stmt
        CALL apoc.nodes.get([stmt[0]]) YIELD node as a
        CALL apoc.nodes.get([stmt[2]]) YIELD node as b
        CALL apoc.create.relationship(a, stmt[1], null, b) YIELD rel
        RETURN rel
        ''' % ','.join(statements)).replace('$label', label))
        return None
