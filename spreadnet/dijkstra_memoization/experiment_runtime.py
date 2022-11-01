# from dijkstra_algorithm import single_source_dijkstra
# from dijkstra_runner import shortest_path

from spreadnet.dijkstra_memoization import dijkstra_runner

# from spreadnet.datasets.data_utils.decoder import pt_decoder

import networkx as nx

# import webdataset as wds
# import timeit
from time import process_time

# import copy
import os
import json


# dataset_path = os.path.join(
#    os.path.dirname(__file__), "..", data_configs["dataset_path"]
# ).replace("\\", "/")

default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "../../experiments/dataset_configs.yaml"
)

dataset_path = os.path.join(os.path.dirname(__file__), "../../experiments/dataset")

# def runTimeTest(graphs, start, end):
#    for g in graphs:
#        #print( nx.single_source_dijkstra(g, start, end) )
#        print( nx.single_source_dijkstra(g, 1, g.number_of_nodes() - 1) )
#    return


def main():
    # G_ten = nx.path_graph(10)
    # G_ten1 = nx.path_graph(20)

    # H = nx.path_graph(10)
    # print( dataset_path )
    # length, path = nx.single_source_dijkstra(G_ten, 0, 1)
    # print('path and length', path, length)
    # TODO: Fixme! this should be an argument.
    f = open(dataset_path + "/raw/random.102.100-110.20.json")
    dataset = json.load(f)
    all_graphs = list()
    # start_nodes = list()
    # end_nodes = list()

    """    {
       "directed": true,
        "multigraph": false,
        "graph": {},
        "nodes": [
            {
                "weight": 1.4213646962011073,
                "pos": [
                    0.8442657485810173,
                    0.8579456176227568
                ],
                "is_end": true,
                "is_start": false,
                "is_in_path": true,
                "id": 0  """

    for d in dataset:
        g = {}
        g["nxdata"] = nx.node_link_graph(d)
        for n in d["nodes"]:
            if n["is_start"]:
                g["start"] = n["id"]
            if n["is_end"]:
                g["end"] = n["id"]
        all_graphs.append(g)

    # print("all_graphs", all_graphs.source)
    print("total_time for ", f)

    print("dijkstra minimum spanning tree with memoization only one path saved")

    for i in range(5):
        t1_start = process_time()
        for g in all_graphs:
            dijkstra_runner.shortest_path_single(g["nxdata"], g["start"], g["end"])
            # print(dijkstra_runner.memoTable)
        # print( "Graph start:", g["start"], " End:", g["end"])
        t1_stop = process_time()
        # print("starttime", t1_start)
        # print("stop time", t1_stop)
        total_time = t1_stop - t1_start
        print(total_time)

    dijkstra_runner.clear_memo_table()
    print("single source dijkstra")

    for i in range(3):
        t1_start = process_time()
        for g in all_graphs:
            nx.single_source_dijkstra(g["nxdata"], g["start"], g["end"])
        # print( "Graph start:", g["start"], " End:", g["end"])
        t1_stop = process_time()
        # print("starttime", t1_start)
        # print("stop time", t1_stop)
        total_time = t1_stop - t1_start
        print(total_time)

    dijkstra_runner.clear_memo_table()
    print("astar")

    for i in range(3):
        t1_start = process_time()
        for g in all_graphs:
            nx.astar_path(g["nxdata"], g["start"], g["end"])
        # print( "Graph start:", g["start"], " End:", g["end"])
        t1_stop = process_time()
        # print("starttime", t1_start)
        # print("stop time", t1_stop)
        total_time = t1_stop - t1_start
        print(total_time)

    dijkstra_runner.clear_memo_table()
    print("dijkstra minimum spanning tree with memoization")

    for i in range(5):
        t1_start = process_time()
        for g in all_graphs:
            dijkstra_runner.shortest_path(g["nxdata"], g["start"], g["end"])
        # print( "Graph start:", g["start"], " End:", g["end"])
        t1_stop = process_time()
        # print("starttime", t1_start)
        # print("stop time", t1_stop)
        total_time = t1_stop - t1_start
        print(total_time)

    dijkstra_runner.clear_memo_table()
    print("Number of graphs: ", len(all_graphs))
    # for g in all_graphs:
    #    print( nx.single_source_dijkstra(g, 1, g.number_of_nodes() - 1) )

    return


if __name__ == "__main__":
    main()
