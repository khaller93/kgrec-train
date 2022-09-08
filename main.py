import typer
from embedding import rdf2vec
from datasets import parse

main = typer.Typer()


@main.command(name='rdf2vec', help='train the rdf2vec embeddings')
def rdf2vec_cmd(epochs: int = 10, walks: int = 200, path_length: int = 4,
                seed: int = 133, model_out_directory: str = 'model',
                dataset: str = 'pokemon'):
    rdf2vec.train(epochs, walks, path_length, seed, model_out_directory,
                  parse(dataset))


@main.command(name='transE')
def trans_e_cmd():
    print('train transE model')


if __name__ == '__main__':
    main()
