Increasing Size experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- run the inc_size_gen_pred.py file first. The first function generates
the experiment_data folder and start to populate it. -The
increasing_graph_size_generator contains the parameters for the graph
generation. The theta and the minimum path length increases as the
number of nodes increases. The rate can be adjusted from the parameters.
This function benefits from having multiple cores as it is parallelized.

- increasing_graph_size_experiment is the part that does the prediction
for each graph. However, this part is not paralelized. Again it creased
it's own folder for the results(increasing_size_predictions).

-  Now we can run the inc_size_analysis.py. The main contains 2 types of
   function calls. Accuracy metric calculators that take the predictions
   and compute different types of accuracy. The second type of function
   call is the plotting that should be moved to the visualization utils.
