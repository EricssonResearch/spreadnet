"""
Really not sure what this thing is.


"""

import collections
import functools
import itertools
from typing import Callable, Optional, Mapping, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import signal
from scipy import spatial
import tensorflow as tf
import tensorflow_gnn as tfgnn



def _set_initial_node_state(cities_set, *, node_set_name):
  assert node_set_name == "cities"
  return tf.concat(
      [tf.cast(cities_set["weight"], tf.float32)[..., None],
       tf.cast(cities_set["is_start"], tf.float32)[..., None],
       tf.cast(cities_set["is_end"], tf.float32)[..., None],
       # Don't provide the position for better generalization.
       # tf.cast(city_features["pos"], tf.float32),
       # Do not give the answer, unless debugging!
       # tf.cast(city_features["is_in_path"], tf.float32)[..., None],
       ],
      axis=-1)

def _set_initial_edge_state(road_set, *, edge_set_name):
  assert edge_set_name == "roads"
  return tf.concat(
      [tf.cast(road_set["weight"], tf.float32)[..., None],
       # Do not give the answer, unless debugging!
       # tf.cast(road_features["is_in_path"], tf.float32)[..., None],
       ],
      axis=-1)

def _set_initial_context_state(context):
  return tfgnn.keras.layers.MakeEmptyFeature()(context)

build_initial_hidden_state = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=_set_initial_node_state,
      edge_sets_fn=_set_initial_edge_state,
      context_fn=_set_initial_context_state)


# Final predictions can often just be read out of the GraphTensor. This colab
# copies them back from the network's output into the input GraphTensor, so that
# consumers of shortest paths (e.g., `draw_graph()`) can be agnostic of whether
# the shortest path is a prediction or a label.
# TODO(b/234563300): Consider supporting `MapFeatures` for multiple graphs,
# to express feature copying with that.

def predict_from_final_hidden_state(task_graph, output_graph):
  """Transforms output logits into "is_in_path" prediction."""

  city_features = task_graph.node_sets["cities"].get_features_dict()
  node_logits = output_graph.node_sets["cities"][
      tfgnn.HIDDEN_STATE]
  city_features["is_in_path"] = tf.cast(
      tf.argmax(node_logits, axis=-1), tf.bool)

  road_features = task_graph.edge_sets["roads"].get_features_dict()
  edge_logits = output_graph.edge_sets["roads"][
      tfgnn.HIDDEN_STATE]
  road_features["is_in_path"] = tf.cast(
      tf.argmax(edge_logits, axis=-1), tf.bool)

  return task_graph.replace_features(
      node_sets={"cities": city_features},
      edge_sets={"roads": road_features})