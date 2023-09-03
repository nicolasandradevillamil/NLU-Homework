# Natural Language Understanding - Advanced Machine Learning 2023-II

In this homework you will learn about the task of language modeling through an implementation of a transformer decoder. You will also learn about [Huggingface](https://huggingface.co/), a community and data science platform with multiple tools for machine learning. We will be using [recipe-nlg-lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite), a smaller version of the [recipe-nlg](https://recipenlg.cs.put.poznan.pl/) dataset. You are going to train models to generate cooking recipes.


## Installation and requirements



## Part 1: Dataset exploration (1 point)

For this part, review the 

**What do the nodes and edges represent? How many are there?**

**What features are used to describe each node?**

**Is the graph directed or undirected?**

**What is the task for this dataset? What metric is used to evaluate this task?**

Then, choose two datasets from the [Open Graph Benchmark](https://ogb.stanford.edu/) datasets that are used for a different task than the Cora dataset and **give a brief description.** 


### Installation:

To download the dataset, run:

wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

This will download a .tgz file containing the relevant files. To unzip this file, run:

tar zxvf cora.tgz

Finally, you must specify the path to the dataset in the `config.json` file inside the `src` directory.

## Part 2: Implementation (1 point)

In this part, you will be implementing the GraphSAGE model for the defined task on the Cora dataset. For this purpose, you should initially dive into the code to get a general understanding of the model. The Cora dataset class is defined in `datasets/node_classification.py` and the model implementation in the rest of the `src/` files. 

a. Based on the Cora dataset class, **explain in your own words the difference between training a transductive and inductive model. What would you expect to give better results?**

b. Also, include in your report an **explanation of the message passing algorithm that is implemented in the forward function in the `model.py` file and the `layers.py` file. How are the messages being aggregated?  How many layers does the model initially have?** Hint: Look to indentify how each of the message passing algorithm steps for a GraphSAGE layer (message calculation for each neighbor, aggregation and activation) is implemented and where. 

To be able to run the code, you have to complete some missing lines in [layers.py](src/layers.py) file. In this file, you will find the different aggregator architectures used in the [GraphSAGE paper](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs). You need to complete the mean aggregator and the pooling aggregator classes. Please only modify the code where you are asked to do so (#TODO).

Once the missing lines are completed, you are ready for the experimentation part!

## Part 3: Experimentation (2 points)

To run the model with the default parameters, go to `src` and use the command: 

```python
python main.py
```

First, experiment with the aggregation arquitecture. 

You must run **4 experiments**, one for each of the available methods.  Then, choose other tunable hyperparameters that you wish to explore and run at leat **6 extra experiments**. In the `src` directory, edit the `config.json` file to specify arguments and flags.

a. To have the complete points, you need to **attach a table with all the experiments to your report.** 

b. In addition, discuss how each of the hyperparameters and aggregating arquitectures that you modified affect the performance of the network.** **Are your results as expected?**

## Part 4: Layer experimentation (1 point)

Finally, you should choose the best model found in Part 4 and experiment with the number of layers in the model. You must perform at least 2 extra experiments. In the report **attach the table with the results and discuss them**.

# Deadline and report

Please upload to your repository a PDF file named Lastname_Graphs.pdf.

Deadline: Sep 11, 2023, 23:59

# References

* [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs), Hamilton et al., NeurIPS 2017.

* This implementation is based on https://github.com/raunakkmr/GraphSAGE

