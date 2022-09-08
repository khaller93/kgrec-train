from enum import Enum


class Dataset(Enum):
    Pokemon = 1

    @property
    def sparql_endpoint(self):
        if self == Dataset.Pokemon:
            return 'http://localhost:7270/repositories/pokemon'
        else:
            raise ValueError('no sparql endpoint known for %s' % self.name)


def parse(dataset: str) -> Dataset:
    if dataset.lower() == Dataset.Pokemon.name.lower():
        return Dataset.Pokemon
    else:
        raise ValueError('the dataset "%s" is unknown' % dataset)
