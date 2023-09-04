# Natural Language Understanding - Advanced Machine Learning 2023-II

In this homework you will learn about the task of language modeling through an implementation of a transformer decoder. You will also learn about [Huggingface](https://huggingface.co/), a community and data science platform with multiple tools for machine learning. We will be using [recipe-nlg-lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite), a smaller version of the [recipe-nlg](https://recipenlg.cs.put.poznan.pl/) dataset. You are going to train models to generate cooking recipes.

# Deadline and report

The grade of this assignment will be determined with the code parts and a written report. In the report you must answer the questions in **bold**. Upload the report to the repository in a PDF file named *Lastname_NLU.pdf*.

## Installation and requirements



## Part 1: Dataset exploration (1 point)

For this part, review the [explore_dataset.py](explore_dataset.py) and [preprocessing.py](preprocessing.py) codes. Observe that [explore_dataset.py](explore_dataset.py) uses the [datasets](https://huggingface.co/docs/datasets/index) library. It loads the dataset directly from the Huggingface hub! Your first task is to explore this dataset that you will use to train the language models. Answer the following questions. You may edit [explore_dataset.py](explore_dataset.py) in order to answer.

**How many recipes does this dataset have in total? How many are in the train and test sets?**

**What columns (features) does the dataset have? Give a brief description of what each one corresponds to.**

Let's find out what kinds of recipes does the dataset have. For this, **determine the 20 most common words used in the recipe names in both the train and test sets. Are they the same for both sets? What happens if you find out the 20 most common words for the recipe steps? Would it be useful information for knowing what kinds of recipes are in the dataset?**

**Estimate the percentage of desserts in the train and test sets.** *Hint: Use the ingredients!*

## Part 2: Micro GPT (1.5 points)

In this part you will use a very small transformer decoder that can generate text trained on the recipes. This will be a baseline language model. 

Check out the [model_baseline.py](model_baseline.py) and [train_baseline.py](train_baseline.py) codes. [model_baseline.py](model_baseline.py) includes the different components necessary in a transformer decoder. Notice that the transformer includes a `block` class that has layer normalizations, multi-head attention and a feedforward layer. The [model_baseline.py](model_baseline.py) code will train this small GPT and output some characters generated.

**Explain the tokenization method that is being used in the model. How big is the vocabulary?**

**How is the positional embedding being implemented? Is it the same as the [original transformers paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)?**

**Explain what the `n_head` `n_embed` and `n_layer` parameters are.**

**Run [train_baseline.py](train_baseline.py). Try 3 different experiments changing hyperparameters. In your report explain what hyperparameters you changed and the perplexity obtained in each experiment.**

**Modify the code in [model_baseline.py](model_baseline.py) so that the transformer becomes an encoder. In the report include an explanation of the changes you made. After that, train the encoder running [train_baseline.py](train_baseline.py). What happens to the loss and perplexity? What is the model generating? Should you use an encoder for language modeling? Why?** Undo the changes so that the transformer is a decoder again. 

**Modify the code in [model_baseline.py](model_baseline.py) removing the residual connections from the transformer. In the report include an explanation of how you removed them. Train the model (decoder) running [train_baseline.py](train_baseline.py). How did the results change? Did they improve? Why?** Undo the changes after answering.

**Complete the final part of [train_baseline.py](train_baseline.py) so that you can add a prompt to the model. After this take your best model from the initial experimentation and perform at least 5 experiments with different prompts. In the report include parts of each text generated from each of your prompts. Try at least 1 prompt completely unrelated to cooking recipes. Analyze how the results qualitatively changed with different prompts.**

### Bonus (0.3 points)
You might have noticed that the perplexity metric is being calculated as the exponential of the cross-entropy loss. However, in class we saw a different definition of perplexity. It turns out these¨definitions are equivalent! **Prove this (mathematically)**. Remember that the formula for perplexity seen in class is:

$Perplexity = \displaystyle{\prod_{t=1}^T(\frac{1}{P_{LM}(x^{t+1}|x^1, x^2, \ldots, x^t)})^\frac{1}{T}}$

## Part 3: Micro GPT with GPT-2 Tokenizer (0.5 points)

The MicroGPT you used in the previous part employs a method of tokenization that is different from the one used in GPT-2. Look at the (train_baseline_newtokens.py)[train_baseline_newtokens.py] code. It is almost the same as [train_baseline.py](train_baseline.py). You are going to modify the tokenizer so that you train the same microGPT model but with new tokens. First complete the final part as you did in [train_baseline.py](train_baseline.py). After that, do the following:

**Complete (train_baseline_newtokens.py)[train_baseline_newtokens.py] with the GPT-2 tokenizer. What is the new vocabulary size?** 

**Additionally, run 3 experiments with this new tokenizer. Include in your report the final perplexities and at least 1 qualitative result.**

## Part 4: GPT-2 (1 point)


## Part 5: Final demo (1 point)

You have now trained a small GPT and GPT-2 for the same task of generating recipes with the same dataset. Take your best microGPT and GPT-2 models and write a demo code in which a user can choose what model to generate from. The user should write a prompt for the model and how many tokens should be generated. The demo should print the result from the model. **In the report you must include the qualitative results of giving the models the exact same prompt. Compare the generated texts from each model.**

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

