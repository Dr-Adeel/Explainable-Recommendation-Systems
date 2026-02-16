from pathlib import Path
import csv
import networkx as nx
import pickle

KG_DIR = Path("data/kg")
TRIPLES = KG_DIR / "triples.csv"
GRAPH_PKL = KG_DIR / "graph.pkl"


def build():
    if not TRIPLES.exists():
        print(f"Missing triples file: {TRIPLES}")
        return

    g = nx.DiGraph()
    with TRIPLES.open("r", encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            s = row.get('subject')
            p = row.get('predicate')
            o = row.get('object')
            if s is None or o is None:
                continue
            # create node for subject and object
            g.add_node(f"item:{s}", type='item')
            # object node may be entity or literal
            g.add_node(f"ent:{o}", type='entity')
            g.add_edge(f"item:{s}", f"ent:{o}", predicate=p)

    with GRAPH_PKL.open("wb") as f:
        pickle.dump(g, f)

    print(f"Built KG with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges; saved to {GRAPH_PKL}")


if __name__ == '__main__':
    build()
