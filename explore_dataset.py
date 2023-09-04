#Use this file to answer the "Explore the dataset" section questions. 
import pandas as pd
from datasets import load_dataset
from preprocessing import create_full_recipes

#Prepare and load dataset
train_dataset = load_dataset("m3hrdadfi/recipe_nlg_lite",split="train")
test_dataset = load_dataset("m3hrdadfi/recipe_nlg_lite",split="test")

#Create a new "Full recipe" column, converting to pandas dataframe
train_dataset = pd.DataFrame(create_full_recipes(train_dataset)) #See preprocessing.py
test_dataset = pd.DataFrame(create_full_recipes(test_dataset))

#TODO 
#1) Find out how many recipes are in the train and test sets
#2) Find out what columns does the dataset have and what each one corresponds to
#3) Determine the 20 most common words in the recipe names for the train and test set
#4) Estimate the percentage of dessert recipes for the train and test sets