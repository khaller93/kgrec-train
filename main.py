import typer
from embedding import rdf2vec, transe, ldsd
from datasets import parse

main = typer.Typer()


@main.command(name='rdf2vec', help='train the rdf2vec embeddings')
def rdf2vec_cmd(epochs: int = 10, walks: int = 200, path_length: int = 4,
                seed: int = 133, model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    rdf2vec.train(parse(dataset), model_out_directory, epochs, walks,
                  path_length, seed)


@main.command(name='transe')
def trans_e_cmd(k: int = 64, epochs: int = 10, batch_size: int = 256,
                seed: int = 133, model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    transe.train(parse(dataset), model_out_directory, k, epochs, batch_size,
                 seed)


@main.command(name='transe-hpo')
def trans_e_cmd(trials: int = 10, model_out_directory: str = 'model',
                dataset: str = 'pokemon', seed: int = 133):
    transe.train_hpo(parse(dataset), model_out_directory, trials=trials,
                     seed=seed)


@main.command(name='ldsd')
def ldsd_cmd(model_out_directory: str = 'model', dataset: str = 'pokemon'):
    ldsd.train(parse(dataset), model_out_directory)


if __name__ == '__main__':
    main()
