import pandas as pd

def create_full_recipes(dataset):
    """
    Creates a new column that includes a full recipe with name, ingredients and steps.
    """
    #You may change this for bonus 2 in part 2.
    fullrecipe= [ " BEGINRECIPE " +
        entry1 + " Ingredients: " + entry2 + " Steps: " + entry3 + " ENDRECIPE "
        for entry1, entry2, entry3 in zip(dataset[:]['name'],dataset[:]['ingredients'],dataset[:]['steps'])
    ]
    #-------------------------------------------
    dataset = dataset.add_column(name="recipe",column=fullrecipe)
    return dataset
    

