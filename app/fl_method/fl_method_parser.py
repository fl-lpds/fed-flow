from app.fl_method import aggregation, clustering, splitting

# a mapping of fl methods to make function call easier
fl_methods = {
    "fed_avg": aggregation.fed_avg,
    "bandwidth": clustering.bandwidth,
    "none_clustering": clustering.none,
    "none_splitting": splitting.none,
    "fake_splitting": splitting.fake,
    "no_splitting": splitting.no_splitting,
    "no_edge_fake_splitting": splitting.no_edge_fake,
    "edge_rl_splitting": splitting.rl_splitting,
    "only_edge_splitting": splitting.only_edge_splitting,
    "only_server_splitting": splitting.only_server_splitting,
    "random_splitting": splitting.randomSplitting,
    "fedmec_splitting": splitting.FedMec,
    # "edge_base_rl_splitting": splitting.edge_based_rl_splitting,
    "edge_based_heuristic": splitting.edge_based_heuristic_splitting
}
