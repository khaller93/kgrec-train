import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON

from kgrec.datasets import Dataset

_load_sparql_limit = 500

_query = """
SELECT ?rB ?p (count(?o1) as ?do) (count(?o2) as ?di) (count(?o3) as ?doi) (count(?o4) as ?dii) WHERE
{
  {
    SELECT ?rB ?p WHERE
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
    ?rA ?p ?rB .
    ?rA ?p ?o1 .
    FILTER (isIRI(?o1)) .
  }
  OPTIONAL {
    ?rB ?p ?rA .
    ?rB ?p ?o2 .
    FILTER (isIRI(?o2)) .
  }
  OPTIONAL {
    ?rA ?p _:u .
    ?rB ?p _:u .
    ?o3 ?p _:v .
    FILTER (isIRI(?o3)) .
  }
  OPTIONAL {
    _:k ?p ?rA .
    _:k ?p ?rB .
    _:l ?p ?o4 .
    FILTER (isIRI(?o4)) .
  }
}
GROUP BY ?rB ?p
ORDER BY ASC(?rB) ASC(?p)
OFFSET %%offset%%
LIMIT %%limit%%
"""


def query_for_ldsd(dataset: Dataset, r_a: str):
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint,
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)

    q = _query.replace('?rA', '<%s>' % r_a, -1)

    offset = 0
    values = {}
    while True:
        if 'pikachu' in r_a:
            print(q.replace(r'%%offset%%', str(offset), 1)
                  .replace('%%limit%%', str(_load_sparql_limit), 1))
        sparql.setQuery(q.replace(r'%%offset%%', str(offset), 1)
                        .replace('%%limit%%', str(_load_sparql_limit), 1))
        ret = sparql.queryAndConvert()

        n = 0
        for r in ret["results"]["bindings"]:
            n += 1
            r_b = r['rB']['value']
            if r_b not in values:
                values[r_b] = {}
            # parse the property values
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

        if n == _load_sparql_limit:
            offset += _load_sparql_limit
        else:
            break

    return values
