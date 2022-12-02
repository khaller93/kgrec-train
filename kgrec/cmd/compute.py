import typer

from sys import stderr
from neo4j import GraphDatabase
from kgrec.datasets import Dataset
from kgrec.metrics.initializer import Initializer
from kgrec.metrics.ldsd import LDSD

app = typer.Typer()
db = typer.Typer()

app.add_typer(db, name='db')


@app.command(name='ldsd', help='compute the LDSD metric')
def compute_ldsd(neo4j_url: str = 'bolt://localhost:7687',
                 neo4j_username: str = 'neo4j', neo4j_password: str = 'neo4j',
                 no_auth: bool = False, model_out_directory: str = 'model',
                 number_of_jobs: int = 1,
                 dataset: str = 'pokemon'):
    auth = (neo4j_username, neo4j_password) if not no_auth else None
    driver = GraphDatabase.driver(neo4j_url, auth=auth)

    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)

    metric = LDSD(dataset=ds, neo4j_driver=driver, num_of_jobs=number_of_jobs)
    metric.compute(model_out_directory)


@db.command(name='init', help='initialize the DB with specified dataset')
def init(dataset: str = 'pokemon', neo4j_url: str = 'bolt://localhost:7687',
         neo4j_username: str = 'neo4j', neo4j_password: str = 'neo4j',
         no_auth: bool = False):
    auth = (neo4j_username, neo4j_password) if not no_auth else None
    driver = GraphDatabase.driver(neo4j_url, auth=auth)

    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)

    initializer = Initializer(dataset=ds, driver=driver)
    initializer.load()


@db.command(name='clear', help='remove all data from the DB')
def clear(dataset: str = 'pokemon', neo4j_url: str = 'bolt://localhost:7687',
          neo4j_username: str = 'neo4j', neo4j_password: str = 'neo4j',
          no_auth: bool = False):
    auth = (neo4j_username, neo4j_password) if not no_auth else None
    driver = GraphDatabase.driver(neo4j_url, auth=auth)

    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)

    initializer = Initializer(dataset=ds, driver=driver)
    initializer.clear()
