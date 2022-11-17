from spreadnet.utils.exp_analysis_utils import AccuracyMetrics

# from multiprocessing import Pool, cpu_count
from spreadnet.utils.visualization_utils import VisualUtils

# import matplotlib.pyplot as plt

# IDEa: we could save the path length also when we do the accuracy.
# Currently for each graph we can deduce the minimum path length.


def prob_accuracy_calls():
    accmet = AccuracyMetrics()
    accmet.prob_accuracy(only_path=False, file_name="all_nodes_acc.csv", use_edges=True)
    accmet.prob_accuracy(only_path=True, file_name="only_path_nodes_acc.csv")
    accmet.prob_accuracy(
        only_path=True,
        file_name="only_path_nodes_acc_no_start_end.csv",
        use_start_end=False,
    )
    accmet.path_length_as_accuracy(file_name="path_length.csv")
    accmet.max_prob_path_lengths(file_name="max_prob_walk.csv")


if __name__ == "__main__":
    vis = VisualUtils()
    # pool = Pool(processes=cpu_count() - 1)
    # pool.starmap(
    #     prob_accuracy, [[False, "all_nodes_acc.csv"],
    #               [True, "only_path_nodes_acc.csv"]]
    # )
    # pool.close()
    # prob_accuracy(only_path=False, file_name="all_nodes_acc.csv")
    # prob_accuracy(only_path=True, file_name="only_path_nodes_acc.csv")
    # max_prob_path_lengths()
    prob_accuracy_calls()

    # vis.prob_plot("max_prob_walk.csv", "Max Prob Walk")
    # # plt.show()
    # vis.prob_plot("path_length_path_length_accuracy.csv", "Path Length Ratio")
    # # plt.show()
    # vis.prob_plot("path_length_percentage_paths.csv", "Percentage paths found")
    # # plt.show()
    # vis.prob_plot("only_path_nodes_acc.csv", "Only Path Nodes")
    vis.prob_plot(
        "only_path_nodes_acc_no_start_end.csv", "Only Path Nodes No Start End"
    )

    # vis.prob_plot("all_nodes_acc.csv", "All Nodes")
    # plt.show()
