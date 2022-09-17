from enum import Enum


class Dataset(Enum):
    Pokemon = 1

    @property
    def sparql_endpoint(self):
        if self == Dataset.Pokemon:
            return 'https://pokemonkg.kevinhaller.dev/sparql/query'
        else:
            raise ValueError('no sparql endpoint known for %s' % self.name)

    @property
    def default_named_graph(self):
        if self == Dataset.Pokemon:
            return 'https://pokemonkg.org/'
        else:
            return None


def parse(dataset: str) -> Dataset:
    if dataset.lower() == Dataset.Pokemon.name.lower():
        return Dataset.Pokemon
    else:
        raise ValueError('the dataset "%s" is unknown' % dataset)
