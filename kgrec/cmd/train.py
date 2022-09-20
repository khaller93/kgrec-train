import typer

from kgrec.datasets import parse
from kgrec.embedding import rdf2vec, transe

app = typer.Typer()


@app.command(name='rdf2vec', help='train rdf2vec embeddings')
def rdf2vec_cmd(epochs: int = 10, walks: int = 200, path_length: int = 4,
                with_reverse: bool = False, skip_type: bool = False,
                seed: int = 133, model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    rdf2vec.train(parse(dataset), model_out_directory, epochs, walks,
                  path_length, with_reverse, skip_type, seed)


@app.command(name='transe', help='train transE embeddings')
def trans_e_cmd(dim: int = 64, scoring_fct_norm: int = 1, epochs: int = 10,
                batch_size: int = 256, loss_margin: float = 1.0,
                negs_per_pos: int = 32, optimizer_lr: float = 0.001,
                seed: int = 133, model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    transe.train(parse(dataset), model_out_directory, dim, scoring_fct_norm,
                 epochs, batch_size, loss_margin, negs_per_pos, optimizer_lr,
                 seed)


@app.command(name='transe-hpo', help='tune parameters and train transE '
                                     'embeddings with best found parameters')
def trans_e_hpo_cmd(trials: int = 10, model_out_directory: str = 'model',
                    dataset: str = 'pokemon', seed: int = 133):
    transe.train_hpo(parse(dataset), model_out_directory, trials=trials,
                     seed=seed)


if __name__ == '__main__':
    app()
