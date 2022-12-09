import time

import typer

from sys import stderr

from kgrec.datasets import Dataset
from kgrec.metrics.graphdb import LocalNeo4JInstance
from kgrec.metrics.ldsd import LDSD

app = typer.Typer()
db = typer.Typer()

app.add_typer(db, name='db')


@app.command(name='ldsd', help='compute the LDSD metric')
def compute_ldsd(dataset: str = 'pokemon', model_out_directory: str = 'model',
                 max_memory: int = 4, number_of_jobs: int = 1):
    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)

    with LocalNeo4JInstance(dataset=ds, mem_in_gb=max_memory,
                            working_dir_path='deployment/neo4j/%s' % ds.name) \
            as inst:
        metric = LDSD(neo4j_details=inst.driver_details, dataset=ds,
                      num_of_jobs=number_of_jobs)
        metric.compute(model_out_directory)


@db.command(name='start', help='compute the LDSD metric')
def start_graph_db(dataset: str = 'pokemon', max_memory: int = 4):
    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)

    with LocalNeo4JInstance(dataset=ds, mem_in_gb=max_memory,
                            working_dir_path='deployment/neo4j/%s' % ds.name) \
            as inst:
        details = inst.driver_details
        print('''Neo4J instance for dataset "%s":
\tBrowser URL:\t\t%s
\tBolt URL:\t\t%s
\tAuthentication:\t\t%s/%s

running... ''' % (ds.name, details.browser_url, details.bolt_url,
                  details.auth[0], details.auth[1]))

        while True:
            time.sleep(10)
