#!/usr/bin/env python
import logging
import requests

from os import makedirs
from os.path import join, dirname, exists
from sys import stdout

url = 'https://kevinhaller.dev/datasets/embeddings'

datasets = ['dbpedia', 'dbpedia100k', 'dbpedia1M', 'pokemon']

if __name__ == '__main__':
    logging.basicConfig(stream=stdout, format='%(levelname)s:\t%(message)s',
                        level=logging.INFO)
    logging.log(logging.INFO, 'Registered a number of datasets: %s' % datasets)
    for ds in datasets:
        logging.log(logging.INFO, 'Check data for dataset: %s' % ds)
        ds_path = join(dirname(__file__), 'data', ds)
        if not exists(ds_path):
            makedirs(ds_path)
        for filename in ['index.tsv.gz', 'relevant_entities.tsv.gz',
                         'statements.tsv.gz']:
            file_path = join(ds_path, filename)
            if not exists(file_path):
                logging.log(logging.INFO,
                            'The file "%s" is downloaded for "%s" ...'
                            % (filename, ds))
                response = requests.get(join(url, ds, filename))
                if 200 < response.status_code < 300:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
            else:
                logging.log(logging.INFO,
                            'The file "%s" was already downloaded for "%s"'
                            % (filename, ds))
