import csv

from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from neo4j import GraphDatabase, Neo4jDriver

from kgrec.datasets import Dataset
from python_on_whales import docker


class LocalNeo4JInstance:
    """ a local instance of Neo4J """

    version = '4.4.4'

    def __init__(self, working_dir_path: str, mem_in_gb: int = 4):
        """
        creates a new local instance of Neo4J with the working directory at the
        specified location.

        :param working_dir_path: path to the working directory for this
        instance.
        :param mem_in_gb: memory dedicated to the Neo4J instance in GB.
        """
        self.pwd = working_dir_path
        self._max_mem = mem_in_gb
        self._container = None

    def _get_volume_dir_paths(self) -> Mapping[str, str]:
        """
        gets a dictionary with the name of volumes as a key, and the path to it
        on the host as a value.

        :return: a dictionary with volume names as key, and corresponding path
        as value.
        """
        return {
            'conf': str(Path(join(self.pwd, 'conf')).absolute()),
            'data': str(Path(join(self.pwd, 'data')).absolute().resolve()),
            'logs': str(Path(join(self.pwd, 'logs')).absolute().resolve()),
            'plugins': str(Path(join(self.pwd,
                                     'plugins')).absolute().resolve()),
            'import': str(Path(join(self.pwd, 'import')).absolute().resolve()),
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

    def create_volume_dirs(self):
        """ creates the directories for the volumes which will be mounted to
        this local Neo4J instance """
        dir_paths = self._get_volume_dir_paths().values()
        for d in dir_paths:
            if not exists(d):
                makedirs(d)

    @property
    def driver(self) -> Neo4jDriver:
        """ gets the Neo4J driver to the local database instance """
        return GraphDatabase.driver('bolt://127.0.0.1:7687',
                                    auth=('neo4j', 'test'))

    def construct_data_importer(self, dataset: Dataset) -> 'DatasetImporter':
        """
        constructs a data importer for the dataset with which the data of this
        specified dataset can be imported into this Neo4J instance.

        :param dataset: for which a data importer shall be constructed.
        :return: the constructed data importer.
        """
        return DatasetImporter(neo4j_instance=self, dataset=dataset)

    def run(self):
        """ runs this local Neo4J instance """
        mm = self._max_mem
        ps = int(mm / 2)
        ct = docker.run(image='neo4j:%s-community' % self.version,
                        detach=True, remove=True,
                        volumes=self._get_volumes(),
                        publish=[
                            ('127.0.0.1:7687', '7687'),
                            ('127.0.0.1:7474', '7474'),
                        ],
                        envs={
                            'NEO$J_AUTH': 'neo4j/test',
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


class DatasetImporter:
    """ an exporter of the dataset in a format that is understood by Neo4J """

    def __init__(self, neo4j_instance: LocalNeo4JInstance, dataset: Dataset):
        self._neo4j_instance = neo4j_instance
        self._dataset = dataset

    def load(self) -> bool:
        """
        loads the given statements into the database with the specified name.

        :return: `True`, if data was imported, otherwise `False`.
        """
        self._neo4j_instance.create_volume_dirs()
        pwd = self._neo4j_instance.pwd
        lock_f = join(pwd, 'init.lock')
        if not exists(lock_f):
            import_dir = join(pwd, 'import')
            self._write_entities(join(import_dir, 'entities.csv'))
            self._write_statements(join(import_dir, 'statements.csv'))
            self._neo4j_instance.execute_load_command()
            with open(lock_f, 'w') as f:
                print('ok', file=f)
            return True
        else:
            return False

    def _write_entities(self, entities_file_path: str):
        """
        writes the entities of the dataset to a CSV file that is understood by
        Neo4J.

        :param entities_file_path: path to the CSV file to which to write
        entities.
        """
        with open(entities_file_path, 'w') as entities_csv_file:
            writer = csv.writer(entities_csv_file, delimiter=',')
            writer.writerow(['tsvID:ID', ':LABEL'])
            cap_name = self._dataset.capitalized_name
            for tsv_id in self._dataset.index.index.values:
                writer.writerow([tsv_id, cap_name])

    def _write_statements(self, statements_file_path: str):
        """
        writes the statements of the dataset to a CSV file that is understood by
        Neo4J.

        :param statements_file_path: path to the CSV file to which to write the
        statements.
        """
        with open(statements_file_path, 'w') as statements_csv_file:
            writer = csv.writer(statements_csv_file, delimiter=',')
            writer.writerow([':START_ID', ':END_ID', ':TYPE'])
            for _, row in self._dataset.statements.iterrows():
                writer.writerow([row['subj'], row['obj'], 'P%d' % row['pred']])


class GraphDB:
    """ Initializer for Neo4J, which loads the statements into the database """

    def __init__(self, neo4j_instance: LocalNeo4JInstance, dataset: Dataset):
        self._neo4j_instance = neo4j_instance
        self._dataset = dataset

    def __enter__(self):
        di = self._neo4j_instance.construct_data_importer(self._dataset)
        di.load()
        self._neo4j_instance.run()
        return self

    @property
    def driver(self) -> Neo4jDriver:
        """ gets the Neo4J driver to the database instance """
        return self._neo4j_instance.driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._neo4j_instance.stop()
