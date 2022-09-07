import typer
from embedding import rdf2vec

main = typer.Typer()

_default_sparql_endpoint = 'http://localhost:7270/repositories/pokemon'


@main.command(name='rdf2vec', help='train the rdf2vec embeddings')
def rdf2vec_cmd(epochs: int = 10, walks: int = 200, path_length: int = 4,
                seed: int = 133,
                sparql_endpoint: str = _default_sparql_endpoint,
                model_out_directory: str = 'model'):
    rdf2vec.train(epochs, walks, path_length, seed, sparql_endpoint,
                  model_out_directory)


@main.command(name='trasnE')
def trans_e_cmd():
    print('train transE model')


if __name__ == '__main__':
    main()
