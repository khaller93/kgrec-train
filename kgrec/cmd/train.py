import typer

from kgrec.datasets import Dataset
from multiprocessing import cpu_count
from sys import stderr

from kgrec.embedding.rdf2vec import RDF2VecModel
from kgrec.embedding.transe import TransEModel, TransEHPO

app = typer.Typer()


@app.command(name='rdf2vec', help='train rdf2vec embeddings')
def rdf2vec_cmd(epochs: int = 10, walks: int = 200, path_length: int = 4,
                with_reverse: bool = False, skip_type: bool = False,
                seed: int = 133, number_of_jobs: int = cpu_count(),
                model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)
    rdf2vec = RDF2VecModel(ds)
    rdf2vec.train(model_kwargs=dict(epochs=epochs, walks=walks,
                                    path_length=path_length,
                                    with_reverse=with_reverse,
                                    skip_type=skip_type, seed=seed),
                  training_kwargs=dict(num_jobs=number_of_jobs))
    rdf2vec.write_to(model_out_directory)


@app.command(name='transe', help='train transE embeddings')
def trans_e_cmd(dim: int = 64, scoring_fct_norm: int = 1, epochs: int = 10,
                batch_size: int = 256, loss_margin: float = 1.0,
                negs_per_pos: int = 32, optimizer_lr: float = 0.001,
                seed: int = 133, model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)
    trans_e = TransEModel(ds)
    trans_e.train(model_kwargs=dict(dim=dim, scoring_fct_norm=scoring_fct_norm,
                                    epochs=epochs, batch_size=batch_size,
                                    loss_margin=loss_margin,
                                    num_negs_per_pos=negs_per_pos,
                                    optimizer_lr=optimizer_lr, seed=seed))
    trans_e.write_to(model_out_directory)


@app.command(name='transe-hpo', help='tune parameters and train transE '
                                     'embeddings with best found parameters')
def trans_e_hpo_cmd(trials: int = 10, model_out_directory: str = 'model',
                    dataset: str = 'pokemon', batch_size: int = None,
                    seed: int = 133):
    ds = Dataset.get_dataset_for(dataset)
    if ds is None:
        print('err: the given dataset "%s" isn\'t supported' % dataset,
              file=stderr)
        exit(1)
    trans_e_hpo = TransEHPO(dataset=ds, model_out_directory=model_out_directory)
    t = trans_e_hpo.hpo(trials=trials, seed=seed, batch_size=batch_size)
    t.write_to(model_out_directory)
