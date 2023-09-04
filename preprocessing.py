import pandas as pd

def create_full_recipes(dataset):
    """
    Creates a new column that includes a full recipe with name, ingredients and steps.
    """
    fullrecipe= [ " BEGINRECIPE " +
        entry1 + " \nIngredients:" + entry2 + " \nSteps: " + entry3 + " ENDRECIPE "
        for entry1, entry2, entry3 in zip(dataset[:]['name'],dataset[:]['ingredients'],dataset[:]['steps'])
    ]
    dataset = dataset.add_column(name="recipe",column=fullrecipe)
    return dataset
    

