import csv
import logging

from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from neo4j import GraphDatabase

from kgrec.datasets import Dataset
from python_on_whales import docker


class Neo4JDetails:
    """ details object for a new Neo4J instance """

    def __init__(self, browser_url: str, bolt_url: str, auth: Tuple[str, str]):
        """
        creates new details object for a new Neo4J instance.

        :param browser_url: URL of the browser interface to the Neo4J instance.
        :param bolt_url: URL of the interface to the Neo4J instance. It must not
        be `None`.
        :param auth: a tuple of username and corresponding password for
        accessing the Neo4J instance. This argument can be `None`, if
        authentication is disabled on the instance.
        """
        self.bolt_url = bolt_url
        self.browser_url = browser_url
        self.auth = auth


_neo4j_local_browser_port = '7474'
_neo4j_local_bolt_port = '7687'
_neo4j_local_passwd = 'test'


class LocalNeo4JInstance:
    """ a local instance of Neo4J """

    version = '4.4.4'

    def __init__(self, dataset: Dataset, working_dir_path: str,
                 mem_in_gb: int = 4):
        """
        creates a new local instance of Neo4J with the working directory at the
        specified location.

        :param dataset: the dataset for which the local instance shall be
        started.
        :param working_dir_path: path to the working directory for this
        instance.
        :param mem_in_gb: memory dedicated to the Neo4J instance in GB.
        """
        self._dataset = dataset
        self._pwd = working_dir_path
        self._max_mem = mem_in_gb
        self._container = None

    def __enter__(self):
        logging.info('aims to start the local Neo4J instance for datasets "%s"',
                     self._dataset.name)
        logging.info('using the directory "%s" for Neo4J instance', self._pwd)
        di = self._construct_data_importer()
        di.load(self._dataset)
        self.run()
        di.index()
        return self

    def _get_volume_dir_paths(self) -> Mapping[str, str]:
        """
        gets a dictionary with the name of volumes as a key, and the path to it
        on the host as a value.

        :return: a dictionary with volume names as key, and corresponding path
        as value.
        """
        return {
            'conf': str(Path(join(self._pwd, 'conf')).absolute()),
            'data': str(Path(join(self._pwd, 'data')).absolute().resolve()),
            'logs': str(Path(join(self._pwd, 'logs')).absolute().resolve()),
            'plugins': str(Path(join(self._pwd,
                                     'plugins')).absolute().resolve()),
            'import': str(Path(join(self._pwd, 'import')).absolute().resolve()),
        }

    def _get_volumes(self) -> Sequence[Tuple[str, str, str]]:
        """ gets the volumes to mount """
        vol_dict = self._get_volume_dir_paths()
        return [
            (vol_dict['data'], '/data', 'rw'),
            (vol_dict['logs'], '/logs', 'rw'),
            (vol_dict['plugins'], '/plugins', 'rw'),
            (vol_dict['conf'], '/conf', 'rw'),
            (vol_dict['import'], '/import', 'ro'),
        ]

    def _construct_data_importer(self) -> 'DatasetImporter':
        """
        constructs a data importer for the dataset with which the data of this
        specified dataset can be imported into this Neo4J instance.

        :return: the constructed data importer.
        """
        import_dir_path = self._get_volume_dir_paths()['import']
        return DatasetImporter(neo4j_instance=self,
                               import_dir_path=import_dir_path)

    def create_volume_dirs(self):
        """ creates the directories for the volumes which will be mounted to
        this local Neo4J instance """
        dir_paths = self._get_volume_dir_paths().values()
        for d in dir_paths:
            if not exists(d):
                makedirs(d)

    @property
    def driver_details(self) -> Neo4JDetails:
        """ gets the Neo4J driver details for this local database instance """
        return Neo4JDetails('http://127.0.0.1:%s' % _neo4j_local_browser_port,
                            'bolt://127.0.0.1:%s' % _neo4j_local_bolt_port,
                            ('neo4j', _neo4j_local_passwd))

    def run(self):
        """ runs this local Neo4J instance """
        mm = self._max_mem
        ps = int(mm / 2)
        ct = docker.run(image='neo4j:%s-community' % self.version,
                        detach=True, remove=True,
                        volumes=self._get_volumes(),
                        publish=[
                            ('127.0.0.1:%s' % _neo4j_local_bolt_port, '7687'),
                            ('127.0.0.1:%s' % _neo4j_local_browser_port,
                             '7474'),
                        ],
                        envs={
                            'NEO4J_AUTH': 'neo4j/%s' % _neo4j_local_passwd,
                            'NEO4J_dbms_memory_pagecache_size': '%dG' % ps,
                            'NEO4J_dbms.memory.heap.initial_size': '%dG' % mm,
                            'NEO4J_dbms_memory_heap_max_size': '%dG' % mm,
                            'NEO4JLABS_PLUGINS': '["apoc"]',
                        })
        self._container = ct

    def execute_load_command(self):
        """ executes a load command on this Neo4J instance """
        docker.run(image='neo4j:%s-community' % self.version, remove=True,
                   volumes=self._get_volumes(),
                   entrypoint='neo4j-admin',
                   command=[
                       'import',
                       '--database', 'neo4j',
                       '--nodes', '/import/entities.csv',
                       '--relationships', '/import/statements.csv',
                       '--id-type', 'INTEGER',
                       '--force',
                   ])

    def stop(self):
        """ stops this local Neo4J instance """
        if self._container is not None:
            docker.container.stop(self._container)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DatasetImporter:
    """ an exporter of the dataset in a format that is understood by Neo4J """

    def __init__(self, import_dir_path: str,
                 neo4j_instance: LocalNeo4JInstance):
        """
        creates a new data importer that writes a given dataset to the specified
        import directory and uses the specified Neo4J instance to which the
        dataset shall be imported.

        :param import_dir_path: to which to write the data in Neo4J format.
        :param neo4j_instance: Neo4J instance to which the dataset shall be
        imported.
        """
        self._pwd = import_dir_path
        self._neo4j_instance = neo4j_instance

    def _get_lock_path(self) -> str:
        """
        gets the path to the lock file.

        :return: path to the lock file.
        """
        return join(self._pwd, 'init.lock')

    def get_entities_path(self) -> str:
        """
        gets the path to the CSV file containing the nodes to import
        into Neo4J.

        :return: path to the CSV file containing the nodes.
        """
        return join(self._pwd, 'entities.csv')

    def get_statements_path(self) -> str:
        """
        gets the path to the CSV file containing the edges to import
        into Neo4J.

        :return: path to the CSV file containing the edges.
        """
        return join(self._pwd, 'statements.csv')

    def load(self, dataset: Dataset) -> bool:
        """
        loads the given statements into the database with the specified name.

        :param dataset: dataset that shall be imported.
        :return: `True`, if data was imported, otherwise `False`.
        """
        self._neo4j_instance.create_volume_dirs()
        lock_f = self._get_lock_path()
        if not exists(lock_f):
            self._write_entities(dataset, self.get_entities_path())
            self._write_statements(dataset, self.get_statements_path())
            self._neo4j_instance.execute_load_command()
            with open(lock_f, 'w') as f:
                print('loaded', file=f)
            return True
        else:
            logging.info('data has already been loaded into Neo4J for "%s"',
                         dataset.name)
            return False

    def index(self) -> bool:
        """ indexes the tsvID property of nodes """
        lock_f = Path(self._get_lock_path())
        if 'indexed' not in lock_f.read_text():
            logging.info(
                'indexing the tsvID property of nodes for faster querying')
            details = self._neo4j_instance.driver_details
            with GraphDatabase.driver(details.bolt_url,
                                      auth=details.auth) as driver:
                with driver.session() as session:
                    session.write_transaction(self._index_unique_tsv_id)
                    with open(str(lock_f.absolute()), 'a') as f:
                        print('indexed', file=f)
                        return True
        return False

    @staticmethod
    def _index_unique_tsv_id(tx):
        tx.run('CREATE INDEX IF NOT EXISTS FOR (n:Resource) ON (n.tsvID)')
        return None

    @staticmethod
    def _write_entities(dataset: Dataset, entities_file_path: str):
        """
        writes the entities of the dataset to a CSV file that is understood by
        Neo4J.

        :param dataset: dataset that shall be imported.
        :param entities_file_path: path to the CSV file to which to write
        entities.
        """
        logging.info(
            'entities from dataset "%s" are written to "%s" in Neo4J format',
            dataset.name, entities_file_path)
        with open(entities_file_path, 'w') as entities_csv_file:
            writer = csv.writer(entities_csv_file, delimiter=',')
            writer.writerow(['tsvID:ID', 'rvKey:int', ':LABEL'])
            rv_df = dataset.relevant_entities
            for tsv_id in dataset.index.index.values:
                rv_ids = rv_df[rv_df['key_index'] == tsv_id].index.values
                writer.writerow([tsv_id, rv_ids[0] if len(rv_ids) == 1 else -1,
                                 'Resource'])

    @staticmethod
    def _write_statements(dataset: Dataset, statements_file_path: str):
        """
        writes the statements of the dataset to a CSV file that is understood by
        Neo4J.

        :param dataset: dataset that shall be imported.
        :param statements_file_path: path to the CSV file to which to write the
        statements.
        """
        logging.info(
            'statements from dataset "%s" are written to "%s" in Neo4J format',
            dataset.name, statements_file_path)
        with open(statements_file_path, 'w') as statements_csv_file:
            writer = csv.writer(statements_csv_file, delimiter=',')
            writer.writerow([':START_ID', ':END_ID', ':TYPE'])
            for triple in dataset.statement_iterator():
                writer.writerow([triple[0], triple[2], 'P%d' % triple[1]])
