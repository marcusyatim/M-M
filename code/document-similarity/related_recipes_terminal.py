'''
Author: Eu Jin Marcus Yatim
This python script asks the user to input a foodnetwork.com recipe url, performs web scrapping on it and and uses the WMD model to match the information to the closest related recipes from the dataset.
Note: Depending if foodnetwork.com changes or modifies their website, this program may or may not work thereafter. Works as of Sep 2021.
This script is to run locally on the terminal.
Requires NLTK packages: stopwords and punkt. Run setup.py to install them.
Requires a 'wmd.model' file and the dataset CSV file.
'''
import gensim
import nltk
import requests

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.similarities import WmdSimilarity

def receive_input():
    '''
    This functions asks for a terminal inputs from the user. I.e. A foodnetwork.com url to a target recipe that the user wants to find related recipes.
    Returns the url to the target recipe as a string.
    '''
    url = input("\nEnter a url to a foodnetwork.com recipe\n- ")
    print ("\nExcellent, you are interested in finding related recipes to:\n" + str(url))

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

def preprocess(list):
    '''
    This function takes in a list of strings (string length can vary and can be >1)
    and tokenises each string in the list. As well as filtering out the tokens based on stopwords, punctuation and numbers.
    Each token is then added to the token_list and the latter is returned.
    '''
    mystopwords = stopwords.words("english")
    WNlemma = nltk.WordNetLemmatizer()
    tokens_list = []
    for item in list:
        tokens = nltk.word_tokenize(item)
        tokens = [ WNlemma.lemmatize(t.lower()) for t in tokens ]
        tokens = [ t for t in tokens if not t in mystopwords ]
        tokens = [ t for t in tokens if t.isalpha() ]   # Remove numbers and punctuation.
        for token in tokens:
            tokens_list.append(token)
    
    return tokens_list

def match_recipes(recipe):
    '''
    This function takes in a target recipe, runs it through a preloaded WMD model and obtains the top 5 closest recipes from the dataset.
    The model outputs the 'steps' of the related recipes. As such, need to match it back to the dataset to get their IDs.
    Returns a list of these recipe ids.
    '''
    wmd_model = WmdSimilarity.load('../../data/document-similarity/wmd.model')
    related_recipes = wmd_model[recipe]
    dataset = pd.read_csv('../../data/RAW_recipes.csv', usecols=['id','steps'])
    steps = dataset['steps']
    related_ids = []
    for i in range(5):
        related_recipe = steps[related_recipes[i][0]]
        related_id = dataset['id'][dataset.index[dataset['steps']==related_recipe].tolist()[0]]
        related_ids.append(related_id)
    
    return related_ids

def recipe_links(ids):
    '''
    This function prints out the Food.com links of the related recipes, based on their ids.
    '''
    print ("\nYour Top 5 related recipes are:")
    for i_d in ids:
        link = "https://www.food.com/recipe/" + str(i_d)
        print (link)

######################
####### MAIN #########
######################
def main():
    # Take in target recipe's url from user
    url = receive_input()

    # Perform web scrapping of the url
    recipe_info = webscrapping(url)

    # Preprocess the target recipe info
    preprocessed_recipe = preprocess(recipe_info)

    # Find related recipes' ids
    related_ids = match_recipes(preprocessed_recipe)

    # Output the recipe link to Food.com
    recipe_links(related_ids)

if __name__ == '__main__':
    main()