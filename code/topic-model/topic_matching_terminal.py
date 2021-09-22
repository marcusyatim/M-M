'''
This python script asks the user to input their likes and dislikes and matches those to the closest recipes.
This script is to run locally on the terminal.
'''
import json
import nltk
import pandas as pd
import string
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

def receive_input():
    '''
    This functions asks for a series of terminal inputs from the user.
    '''
    i = 1
    likes = []
    dislikes = []
    while i:
        if i == 1:
            like = input("\nWhat type of recipe would you like today? (E.g. quick/easy)\n- ")
            likes.append(like.title())
            i += 1
        like = input("Anything else? Type 'Done' when you are finished.\n- ")
        if like.title() == 'Done':
            break
        likes.append(like.title())
    print ("\nExcellent, you are interested in the following:\n" + str(likes))
    while i:
        if i == 2:
            dislike = input("\nIs there anything you don't want to see in a recipe? (E.g. spicy) (Type 'Done' to skip)\n- ")
            if dislike.title() == 'Done':
                print ("\nGotcha.")
                break
            dislikes.append(dislike.title())
            i += 1
        dislike = input("Anything else? Type 'Done' when you are finished.\n- ")
        if dislike.title() == 'Done':
            break
        dislikes.append(dislike.title())
    if len(dislikes) != 0:
        print ("\nGotcha, so you would like to avoid the following:\n" + str(dislikes))

    return likes, dislikes

def preprocess(list):
    '''
    This function takes in a list of strings (string length can vary and can be >1)
    and tokenises each string in the list. As well as filtering out the tokens based on stopwords, punctuation and length.
    Each token is then added to the token_list and the latter is returned.
    '''
    mystopwords = stopwords.words("english")
    WNlemma = nltk.WordNetLemmatizer()
    tokens_list = []
    for item in list:
        tokens = nltk.word_tokenize(item)
        tokens = [ t for t in tokens if t not in string.punctuation+"’“”'" ]
        tokens = [ WNlemma.lemmatize(t.lower()) for t in tokens ]
        tokens = [ t for t in tokens if t not in mystopwords ]
        tokens = [ t for t in tokens if len(t) >= 3 ]
        for token in tokens:
            tokens_list.append(token)

    return tokens_list

def wordnet(list):
    '''
    This function takes in a list of tokens and gets the synonyms, hyponyms, hypernyms, meronyms, holonyms & entailments for each token.
    Duplicates are avoided by using set().
    Returns them as a set.
    '''
    wordnet = set()
    for token in list:
        for synset in wn.synsets(token):
            for lemma in synset.lemmas():
                wordnet.add(lemma.name())
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemma_names():
                    wordnet.add(lemma)
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemma_names():
                    wordnet.add(lemma)
            for meronym in synset.part_meronyms():
                for lemma in meronym.lemma_names():
                    wordnet.add(lemma)
            for holonym in synset.part_holonyms():
                for lemma in holonym.lemma_names():
                    wordnet.add(lemma)
            for entailment in synset.entailments():
                for lemma in entailment.lemma_names():
                    wordnet.add(lemma)

    return wordnet 

def match_topics(likes, dislikes, topic_dict):
    '''
    This function uses a simple score based system. 
    If the word in likes is found in a topic, score +1 for that topic.
    If the word in dislikes is found in a topic, score -1 for that topic.
    Scores are saved in a dictionary with the topics as keys and scores as values. If score for a topic is <1, do not consider that topic.
    If no topics have score >0, return error message and exit. Else return a sorted top 3 highest scoring topics as a list.
    '''
    score_dict = {}
    for key, value in topic_dict.items():
        score = 0
        for like in likes:
            if like in value:
                score += 1
        for dislike in dislikes:
            if dislike in value:
                score -= 1
        if score < 1:
            continue
        else:
            score_dict[key] = score
    if not score_dict:
        print ("\nYour search did not produce any results. Kindly try again with a more specific search.")
        quit()
    else:
        likely_topics = sorted(score_dict, key=score_dict.get, reverse=True)[:3]

    return likely_topics

def match_recipes(exp, topics):
    '''
    This function sorts the recipes based on the likely topics (up to 3) and with priority based on the topics ranked order.
    After sorting, extract the ids of the top 5 recipes. Returns a list of these recipe ids.
    '''
    likely_recipes = []
    recipes = pd.read_csv('../../data/topic-model/assignments/' + exp + '.csv', usecols=['id'] + topics)

    # This line shuffles the dataframe. This is to create randomness as many recipes have the same probabilities.
    recipes = recipes.sample(frac=1)

    if len(topics) == 1:
        recipes = recipes.sort_values(topics[0], ascending=False)
    elif len(topics) == 2:
        recipes = recipes.sort_values([topics[0], topics[1]], ascending=False)
    else:
        recipes = recipes.sort_values([topics[0], topics[1], topics[2]], ascending=False)
    recipes = recipes['id'].head(5)
    for recipe in recipes:
        likely_recipes.append(recipe)
    
    return likely_recipes

def recipe_links(recipes):
    '''
    This function prints out the Food.com links of the likely recipes, based on their ids.
    '''
    print ("\nYour Top 5 recommended recipes are:")
    for recipe in recipes:
        link = "https://www.food.com/recipe/" + str(recipe)
        print (link)

######################
####### MAIN #########
######################
def main():
    # Define experiment to be used
    exp = 'exp3'

    # Take in likes and dislikes from user
    likes, dislikes = receive_input()

    # Preprocess the likes and dislikes (remove stopwords, punctuations, tokenisation)
    likes_tokens = preprocess(likes)
    dislikes_tokens = preprocess(dislikes)

    # Get wordnet of likes and dislikes
    likes_wordnet = wordnet(likes_tokens)
    dislikes_wordnet = wordnet(dislikes_tokens)

    # Get topic dictionary
    topic_dict = json.load(open('../../data/topic-model/topN/' + exp + '.json'))

    # Match likes and dislikes to the most likely topic(s)
    likely_topics = match_topics(likes_wordnet, dislikes_wordnet, topic_dict)

    # Match topic with recipes
    likely_recipes = match_recipes(exp, likely_topics)

    # Output the recipe link to Food.com
    recipe_links(likely_recipes)

if __name__ == '__main__':
    main()