import progressbar as pb

from neo4j import Neo4jDriver
from kgrec.datasets import Dataset


class Initializer:
    """ Initializer for Neo4J, which loads the statements into the database """

    def __init__(self, dataset: Dataset, driver: Neo4jDriver):
        self._dataset = dataset
        self._driver = driver

    def load(self):
        """ loads the given statements into the database with the specified name
        """
        nodes = {}
        with self._driver.session() as session:
            with pb.ProgressBar(max_value=len(self._dataset.statements)) as p:
                for i, row in self._dataset.statements.iterrows():
                    subj = int(row['subj'])
                    obj = int(row['obj'])
                    pred = int(row['pred'])
                    if subj not in nodes:
                        n = session.write_transaction(self._create_node,
                                                      self._dataset.capitalized_name,
                                                      subj)
                        nodes[subj] = n
                    if obj not in nodes:
                        n = session.write_transaction(self._create_node,
                                                      self._dataset.capitalized_name,
                                                      obj)
                        nodes[obj] = n
                    session.write_transaction(self._create_edge, subj, obj,
                                              pred)
                    p.update(int(i) + 1)

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
    def _create_node(tx, label: str, node_id: int):
        result = tx.run('CREATE (n:Resource:%s) SET n.tsv_id = $id RETURN n' %
                        label, id=node_id)
        return result.single()[0]

    @staticmethod
    def _create_edge(tx, a_id: int, b_id: int, prop_id: int):
        result = tx.run('''
        MATCH (a:Resource), (b:Resource)
        WHERE a.tsv_id = $a_id and b.tsv_id = $b_id
        CREATE (a)-[r:p%d]->(b)
        RETURN type(r)
        ''' % prop_id, a_id=a_id, b_id=b_id)
        return result.single()
