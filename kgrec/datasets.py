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
    def default_graph(self):
        return None

    @property
    def ignore_named_graphs(self):
        if self == Dataset.Pokemon:
            return ['http://www.openlinksw.com/schemas/virtrdf#',
                    'http://localhost:8890/sparql',
                    'http://localhost:8890/DAV/']
        else:
            return None


def parse(dataset: str) -> Dataset:
    if dataset.lower() == Dataset.Pokemon.name.lower():
        return Dataset.Pokemon
    else:
        raise ValueError('the dataset "%s" is unknown' % dataset)
