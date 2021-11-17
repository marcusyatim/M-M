import time

from celery import Celery
from datetime import datetime, timedelta
from flask import Flask, request

from getRatings_chatbot import getRatings
from getRecommendations_chatbot import getRecommendations
from getTags_chatbot import getTags

# Initialise the flask app
app = Flask(__name__)
app.config.from_object("config")

# Initialise Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Define a celery asynchronous task
@celery.task
def celery_getRatings(review):
    rating = getRatings(review)
    return rating

# Create a route for the asynchronous task
@app.route('/background_getRatings', methods=['POST'])
def background_getRatings(review):
    rating = celery_getRatings.apply_async(args=[review])
    return rating

# Create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    global likes, dislikes, targetTags, review, rating

    req = request.get_json(silent=True, force=True)
    fulfillmentText = ''
    followupEvent = ''
    query_result = req.get('queryResult')

    # Start
    if query_result.get('action') == 'start':
        fulfillmentText = "Hi, welcome.\nThere are three tasks that this chatbot can run. Type their corresponding names or numbers, as shown below, to run a task!\n(1) Get recommendations\n(2) Get tags\n(3) Get ratings\nTo get back to this message, just type 'start'."    
    
    # Get recommendations
    elif query_result.get('action') == 'get.recommendations':
        fulfillmentText = "What type of recipe would you like today?"
    elif query_result.get('action') == 'get.recommendations.likes':
        likes = str(query_result.get('parameters').get('likes'))
        print (likes)
        fulfillmentText = "Is there anything you don't want to see in a recipe? (E.g. spicy) (Type 'No' to skip)"
    elif query_result.get('action') == 'get.recommendations.dislikes':
        dislikes = str(query_result.get('parameters').get('dislikes'))
        print (dislikes)
        links = getRecommendations(likes, dislikes)
        fulfillmentText = links
    
    # Get tags
    elif query_result.get('action') == 'get.tags':
        fulfillmentText = "Enter a URL to a foodnetwork.com recipe to get its tags."
    elif query_result.get('action') == 'get.tags.url':
        url = str(query_result.get('parameters').get('url'))
        print (url)
        tags = getTags(url)
        fulfillmentText = tags

    # Get ratings
    elif query_result.get('action') == 'get.ratings':
        fulfillmentText = "Enter your review to generate its rating (process will take a while)."
    elif query_result.get('action') == 'get.ratings.review':
        review = str(query_result.get('parameters').get('review'))
        print (review)
        rating = background_getRatings(review)

        # Delay time by 4 secs (Dialogflow webhook will timeout after 5 secs)
        time.sleep(4)
        print (rating.state)
        if rating.state == 'PENDING':
            followupEvent = 'pending'
        elif rating.state == 'SUCCESS':
            fulfillmentText = "Based on your review, the rating (from 1-5) of the recipe is: " + str(rating.info)
    elif query_result.get('action') == 'get.ratings.review.pending1':
        time.sleep(4)
        print (rating.state)
        if rating.state == 'PENDING':
            followupEvent = 'pending'
        elif rating.state == 'SUCCESS':
            fulfillmentText = "Based on your review, the rating (from 1-5) of the recipe is: " + str(rating.info)
    elif query_result.get('action') == 'get.ratings.review.pending2':
        time.sleep(4)
        print (rating.state)
        if rating.state == 'PENDING':
            fulfillmentText = "Timeout error"
        elif rating.state == 'SUCCESS':
            fulfillmentText = "Based on your review, the rating (from 1-5) of the recipe is: " + str(rating.info)        

    return {
            "fulfillmentText": fulfillmentText,
            "source": "webhookdata",
            "followupEventInput": {
                "name": followupEvent
            }
        }

# Run the app
if __name__ == '__main__':
    app.run(debug=True)