# Natural Language Understanding - Advanced Machine Learning 2023-II

In this homework you will learn about the task of language modeling through an implementation of a transformer decoder. You will also learn about [Huggingface](https://huggingface.co/), a community and data science platform with multiple tools for machine learning. We will be using [recipe-nlg-lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite), a smaller version of the [recipe-nlg](https://recipenlg.cs.put.poznan.pl/) dataset. You are going to train models to generate cooking recipes.

# Deadline and report

The grade of this assignment will be determined with the code parts and a written report. In the report you must answer the questions in **bold**. Upload the report to the repository in a PDF file named *Lastname_NLU.pdf*.

## Installation and requirements


## Part 1: Dataset exploration (1 point)

For this part, review the [explore_dataset.py](explore_dataset.py) and [preprocessing.py](preprocessing.py) codes. Observe that [explore_dataset.py](explore_dataset.py) uses the [datasets](https://huggingface.co/docs/datasets/index) library. It loads the dataset directly from the Huggingface hub! Your first task is to explore this dataset that you will use to train the language models. Answer the following questions. You may edit [explore_dataset.py](explore_dataset.py) in order to answer.

**How many recipes does this dataset have in total? How many are in the train and test sets? (0.2 points)**

**What columns (features) does the dataset have? Give a brief description of what each one corresponds to. (0.2 points)**

Let's find out what kinds of recipes does the dataset have. For this, **determine the 20 most common words used in the recipe names in both the train and test sets. Are they the same for both sets? What happens if you find out the 20 most common words for the recipe steps? Would it be useful information for knowing what kinds of recipes are in the dataset? (0.4 points)**

**Estimate the percentage of desserts in the train and test sets. (0.2 points)** *Hint: Use the ingredients!*

## Part 2: Micro GPT (2 points)

In this part you will use a very small transformer decoder that can generate text trained on the recipes. This will be a baseline language model. 

Check out the [model_baseline.py](model_baseline.py) and [train_baseline.py](train_baseline.py) codes. [model_baseline.py](model_baseline.py) includes the different components necessary in a transformer decoder. Notice that the transformer includes a `block` class that has layer normalizations, multi-head attention and a feedforward layer. The [model_baseline.py](model_baseline.py) code will train this small GPT and output some characters generated.

**Explain the tokenization method that is being used in the model. How big is the vocabulary? (0.2 points)**

**How is the positional embedding being implemented? Is it the same as the [original transformers paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (0.2 points)?**

**Explain what the `n_head` `n_embed` and `n_layer` parameters are. (0.1 points)**

**Is the model using the same ´`n_head` and `n_layer` parameters as the [the original GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)? (0.2 points)**

Run [train_baseline.py](train_baseline.py). Choose two hyperparameters of the model architecture and do a 3x3 grid search, this means you have to choose 3 values for each of the 2 hyperparameters and try out all 9 possible combinations. **In your report explain what hyperparameters you changed. Include a table with the final perplexity obtained in the 3x3 grid search. (0.5 points)**

**Modify the code in [model_baseline.py](model_baseline.py) so that the transformer becomes an encoder. In the report include an explanation of the changes you made. After that, train the encoder running [train_baseline.py](train_baseline.py). What happens to the loss and perplexity? What is the model generating? Should you use an encoder for language modeling? Why? (0.3 points)** Undo the changes so that the transformer is a decoder again. Don't forget to include the hyperparameters you used for training the encoder. 

**Modify the code in [model_baseline.py](model_baseline.py) removing the residual connections from the transformer. In the report include an explanation of how you removed them. Train the model (decoder) running [train_baseline.py](train_baseline.py). How did the results change? Did they improve? Why? (0.2 points)** Undo the changes after answering. Again, don't forget to mention the hyperparameters you used for training this model.

**Complete the final part of [train_baseline.py](train_baseline.py) so that you can add a prompt to the model. After this take your best model from the initial experimentation and perform at least 5 experiments with different prompts. In the report include parts of each text generated from each of your prompts. Try at least 1 prompt completely unrelated to cooking recipes. Analyze how the results qualitatively changed with different prompts. (0.3 points)**

Check out the original [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Bonus 1 (0.3 points)
You might have noticed that the perplexity metric is being calculated as the exponential of the cross-entropy loss. However, in class we saw a different definition of perplexity. It turns out these definitions are equivalent! **Prove this (mathematically)**. Remember that the formula for perplexity seen in class is:

$Perplexity = \displaystyle{\prod_{t=1}^T(\frac{1}{P_{LM}(x^{t+1}|x^1, x^2, \ldots, x^t)})^\frac{1}{T}}$

### Bonus 2 (0.1 points)
In [preprocessing.py](preprocessing.py) a full recipe is being created in a very specific way, with the following format:

```
" BEGINRECIPE " + entry1 + " Ingredients: " + entry2 + " Steps: " + entry3 + " ENDRECIPE "
```

Try out your own way for creating full recipes. **Include in the report a comparison between the final perplexity obtained with the original format and with your format.** Keep the same hyperparameters for this comparison. **Explain whether one of the formats for the full recipe would help the model in the generation task and why.**

## Part 3: Micro GPT with GPT-2 Tokenizer (0.5 points)

The MicroGPT you used in the previous part employs a method of tokenization that is different from the one used in GPT-2. Look at the [train_baseline_newtokens.py](train_baseline_newtokens.py) code. It is almost the same as [train_baseline.py](train_baseline.py). You are going to modify the tokenizer so that you train the same microGPT model but with new tokens. First complete the final part as you did in [train_baseline.py](train_baseline.py). After that, do the following:

**Complete [train_baseline_newtokens.py](train_baseline_newtokens.py) with the GPT-2 tokenizer. What is the new vocabulary size?** 

**Additionally, run 3 experiments with this new tokenizer. Include in your report the final perplexities and at least 1 qualitative result.**

## Part 4: GPT-2 (0.5 points)

Now you are going to finetune a pretrained GPT-2 for generating recipes. Analyze [train_GPT2.py](train_GPT2.py). This code uses more libraries from Huggingface. It includes [Transformers](https://huggingface.co/docs/transformers/index) and [Accelerate](https://huggingface.co/docs/accelerate/index).

**Run [train_GPT2.py](train_GPT2.py). Try at least 2 experiments changing hyperparameters. Include in your report the final perplexity and at least 1 qualitative result.** Note: This will take considerably longer time than training the previous models. 

**Experiment with the same prompts you used in part 2. Analyze how the results changed with this new model. Is GPT-2 performing better than the previous model?**

## Part 5: Demo (0.5 points)

You have now trained a small GPT and finetuned GPT-2 for the same task of generating recipes with the same dataset. Take your best microGPT and GPT-2 models and write a demo code in which a user can choose what model to generate from. The user should write a prompt for the model and how many tokens should be generated. The demo should print the result from the model. Name the code `demo.py` and upload it to the repository. The code should work so that if the user runs the command `python demo.py` in the terminal, then a text is printed asking what model should be used. The user then answers and the code asks for a number of tokens generated. After this, the code should ask for the prompt. Finally, it should print the output from the mmodel **In the report you must include the qualitative results of giving the models the exact same prompt. Compare the generated texts from each model. (0.5 points)**

## Part 6: Conclusions (0.5)

Write in your report an analysis of your overall results. It must include answers to the following questions:
**What would you modify for the models to generate better recipes?**
**Was the size of the dataset a problem for training the models? Why?**

# References

* The microGPT implementation is based on https://github.com/karpathy/ng-video-lecture.
* The GPT-2 implementation is based on https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt

