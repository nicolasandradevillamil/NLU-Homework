import pandas as pd

def create_full_recipes(dataset):
    """
    Creates a new column that includes a full recipe with name, ingredients and steps.
    """
    fullrecipe= [
        entry1 + " " + entry2 + " " + entry3
        for entry1, entry2, entry3 in zip(dataset[:]['name'],dataset[:]['ingredients'],dataset[:]['steps'])
    ]
    dataset = dataset.add_column(name="recipe",column=fullrecipe)
    dataset = pd.DataFrame(dataset)
    return dataset
    

