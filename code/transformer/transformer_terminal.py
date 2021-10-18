'''
Author: Eu Jin Marcus Yatim
This python script asks the user to input a foodnetwork.com recipe url, performs web scrapping on it and inputs the data into the transformer to return tags for the recipe.
This script is to run locally on the terminal.
Requires run_transformer.py
'''
import run_transformer
import requests

from bs4 import BeautifulSoup

def receive_input():
    '''
    This functions asks for a terminal input from the user. I.e. A foodnetwork.com url to a target recipe that the user wants to convert to tags.
    Returns the url to the target recipe as a string.
    '''
    url = input("\nEnter a url to a foodnetwork.com recipe\n- ")
    print ("\nThe recipe from " + str(url) + " will be converted to tags:")

    return url

def webscrapping(url):
    '''
    This function performs web scraping on a foodnetwork.com recipe.
    Returns the recipe information as a list.
    '''
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all('li', attrs={'class': 'o-Method__m-Step'})
    recipe_info = []
    for result in results:
      recipe_info.append(result.text.strip())

    return recipe_info

def preprocess(sentence):
    '''
    This function takes in a string and removes certain punctuations from it.
    It will also append the start and end token to the string,
    Returns the preprocessed string.
    '''

    # Strip "[]'," from the sentence
    sentence = sentence.translate(str.maketrans('', '', ".[]',"))

    # Adding a start and an end token to the sentence so that the model know when to start and stop predicting.
    sentence = '<start> ' + sentence + ' <end>'

    return sentence

######################
####### MAIN #########
######################
def main():
    # Take in target recipe's url from user
    url = receive_input()

    # Perform web scrapping of the url
    recipe_info = webscrapping(url)

    # Preprocess the target recipe info
    preprocessed_recipe = preprocess(str(recipe_info))

    # Get tags
    tags = run_transformer.getTags(preprocessed_recipe)

    print ("\n" + tags)

if __name__ == '__main__':
    main()