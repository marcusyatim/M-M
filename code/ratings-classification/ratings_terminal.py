'''
Author: Eu Jin Marcus Yatim
This python script asks the user to input a review for a recipe. The script will then automatically give a rating (from 0-5) of the recipe based on the review.
This script is to run locally on the terminal.
Requires run_BERT.py
'''
import run_BERT

def receive_input():
    '''
    This functions asks for a terminal input from the user. I.e. A review for a recipe.
    Returns the review.
    '''
    review = input("\nEnter your review\n- ")

    return review

######################
####### MAIN #########
######################
def main():
    # Take in review of recipe
    review = receive_input()
    
    # Run BERT to get the rating of the review
    rating = run_BERT.get_results(review)

    print ("\nBased on your review, the rating (from 0-5) of the recipe is: " + str(rating))

if __name__ == '__main__':
    main()