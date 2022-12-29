import json
from networkx.readwrite import json_graph


def tsv2json(input_file, output_file):
    arr = []
    file = open(input_file, "r")
    a = file.readline()

    # The first line consist of headings of the record
    # so we will store it in an array and move to
    # next line in input_file.
    titles = [t.strip() for t in a.split("\t")]
    for line in file:
        d = {}
        for t, f in zip(titles, line.split("\t")):

            # Convert each row into dictionary with keys as titles
            d[t] = f.strip()

        # we will use strip to remove '\n'.
        arr.append(d)

        # we will append all the individual dictionaires into list
        # and dump into file.
    with open(output_file, "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(arr, indent=4))


# Driver Code
input_filename = "xyz.tsv"
output_filename = "xyz.json"
tsv2json(input_filename, output_filename)


# convert json to networkX


def save_json(filename, graph):
    g = graph
    g_json = json_graph.node_link_data(g)
    json.dump(g_json, open(filename, "w"), indent=2)


# def read_json_file(filename):
#   graph = json_graph.loads(open(filename))
#    return graph

# since the json file is written using json.
# dump, then use json.load to read the contents back.
def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.loads(json.load(f))
    return json_graph.nx.node_link_graph(js_graph)
