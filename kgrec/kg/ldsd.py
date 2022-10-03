import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON

from kgrec.datasets import Dataset

_load_sparql_limit = 10000

_query = """
SELECT ?rB ?p ?do ?di ?dio ?dii WHERE
{
  {
    SELECT distinct ?rB ?p WHERE
    {
      {?rA ?p ?rB}
      UNION
      {?rB ?p ?rA}
      UNION
      {
        ?rA ?p _:x .
        ?rB ?p _:x .
      }
      UNION
      {
        _:y ?p ?rA .
        _:y ?p ?rB .
      }
      FILTER(isIRI(?rB) && ?rB != ?rA) .
    }
  }
  OPTIONAL {
    SELECT ?rB ?p (count(?o1) as ?do) WHERE
    {
        ?rA ?p ?rB .
        ?rA ?p ?o1 .
        FILTER (isIRI(?o1) && ?rB != ?rA) .
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?rB ?p (count(?o2) as ?di) WHERE
    {
        ?rB ?p ?rA .
        ?rB ?p ?o2 .
        FILTER (isIRI(?o2) && ?rB != ?rA) .
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?rB ?p (count(?o3) as ?dio) WHERE
    {
        ?rA ?p _:u .
        ?rB ?p _:u .
        ?rA ?p _:v .
        ?o3 ?p _:v .
        FILTER (isIRI(?o3) && ?rB != ?rA) .
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?rB ?p (count(?o4) as ?dii) WHERE
    {
        _:k ?p ?rA .
        _:k ?p ?rB .
        _:l ?p ?rA .
        _:l ?p ?o4 .
        FILTER (isIRI(?o4) && ?rB != ?rA) .
    }
    GROUP BY ?rB ?p
  }
}
ORDER BY ASC(?rB) ASC(?p)
OFFSET %%offset%%
LIMIT %%limit%%
"""


def query_for_ldsd(dataset: Dataset, r_a: str):
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint + '/query',
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)

    q = _query.replace('?rA', '<%s>' % r_a, -1)

    offset = 0
    values = {}
    while True:
        sparql.setQuery(q.replace('%%offset%%', str(offset), 1)
                        .replace('%%limit%%', str(_load_sparql_limit), 1))
        ret = sparql.queryAndConvert()

        n = 0
        for r in ret["results"]["bindings"]:
            r_b = r['rB']['value']
            if r_b not in values:
                values[r_b] = {}
            p = r['p']['value']
            di = np.float64(r['di']['value']) if 'di' in r else None
            do = np.float64(r['do']['value']) if 'do' in r else None
            dio = np.float64(r['dio']['value']) if 'dio' in r else None
            dii = np.float64(r['dii']['value']) if 'dii' in r else None
            values[r_b][p] = {
                'di': di,
                'do': do,
                'dio': dio,
                'dii': dii,
            }
            n += 1

        if n == _load_sparql_limit:
            offset += _load_sparql_limit
        else:
            break

    return values
