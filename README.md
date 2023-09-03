# Natural Language Understanding - Advanced Machine Learning 2023-II

In this homework you will learn about the task of language modeling through an implementation of a transformer decoder. You will also learn about [Huggingface](https://huggingface.co/), a community and data science platform with multiple tools for machine learning. We will be using [recipe-nlg-lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite), a smaller version of the [recipe-nlg](https://recipenlg.cs.put.poznan.pl/) dataset. You are going to train models to generate cooking recipes.

# Deadline and report

The grade of this assignment will be determined with the code parts and a written report. In the report you must answer the questions in **bold**. Upload the report to the repository in a PDF file named *Lastname_NLU.pdf*.

## Installation and requirements



## Part 1: Dataset exploration (1 point)

For this part, review the [explore_dataset.py](explore_dataset.py) and [preprocessing.py](preprocessing.py) codes. Observe that [explore_dataset.py](explore_dataset.py) uses the [datasets](https://huggingface.co/docs/datasets/index) library. It loads the dataset directly from the Huggingface hub! Your first task is to explore this dataset that you will use to train the language models. Answer the following questions. You may edit [explore_dataset.py](explore_dataset.py) in order to answer.

**How many recipes does this dataset have in total? How many are in the train and test sets?**

**What columns (features) does the dataset have? Give a brief description of what each one corresponds to.** Notice that the function *create_full_recipes* outputs a pandas DataFrame.

Let's find out what kinds of recipes does the dataset have. For this, **determine the 20 most common words used in the recipe names in both the train and test sets. Are they the same for both sets? What happens if you find out the 20 most common words for the recipe steps? Would it be useful information for knowing what kinds of recipes are in the dataset?**

**Estimate the percentage of desserts in the train and test sets.** *Hint: Use the ingredients!*

## Part 2: Micro GPT

In this part, you will complete a very small decoder transformer that can generate text trained on the recipes. 

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

