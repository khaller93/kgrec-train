import gzip

import requests
from rdflib import Graph, Literal

if __name__ == '__main__':
    print('downloading statements ...')
    response = requests.get('https://kevinhaller.dev/datasets/dbpedia/100k/22-03-dbpedia_all.nt.gz')
    print('decompressing statements ...')
    nt = gzip.decompress(response.content)
    print('parsing statements ...')
    g = Graph()
    g.parse(nt)
    with open('dbpedia100k.tsv', 'w+') as f:
        for sub, p, obj in g:
            if isinstance(obj, Literal):
                continue
            f.write('%s\t%s\t%s\n' % (str(sub), str(p), str(obj)))
