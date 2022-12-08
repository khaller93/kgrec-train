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
                 number_of_jobs: int = 1):
    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)

    with LocalNeo4JInstance(dataset=ds, working_dir_path='deployment/neo4j/%s'
                                                         % ds.name) as inst:
        metric = LDSD(neo4j_details=inst.driver_details, dataset=ds,
                      num_of_jobs=number_of_jobs)
        metric.compute(model_out_directory)
