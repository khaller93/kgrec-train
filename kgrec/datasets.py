from enum import Enum


class Dataset(Enum):
    Pokemon = 1
    DBpedia100k = 2

    @property
    def sparql_endpoint(self):
        if self == Dataset.Pokemon:
            return 'https://pokemonkg.kevinhaller.dev/sparql'
        elif self == Dataset.DBpedia100k:
            return 'https://dbpedia100k.kevinhaller.dev/sparql'
        else:
            raise ValueError('no sparql endpoint known for %s' % self.name)

    @property
    def default_graph(self):
        return None

    @property
    def ignore_named_graphs(self):
        return None

    @property
    def statements_file(self):
        if self == Dataset.DBpedia100k:
            return '22-03-dbpedia-all.tsv.gz', \
                   'https://kevinhaller.dev/datasets/dbpedia/100k/22-03-dbpedia-all.tsv.gz'
        else:
            return None


def parse(dataset: str) -> Dataset:
    if dataset.lower() == Dataset.Pokemon.name.lower():
        return Dataset.Pokemon
    elif dataset.lower() == Dataset.DBpedia100k.name.lower():
        return Dataset.DBpedia100k
    else:
        raise ValueError('the dataset "%s" is unknown' % dataset)
