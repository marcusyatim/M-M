# Marcus Y Food

The source code and data in this repo are meant for running locally as a testing environment.

For deployment to production on Heroku with Docker, refer to the accompanying repo https://github.com/marcusyatim/M-M-deploy. 

## Setup

Some large files in data folder. Suggest to install `git lfs` @ https://git-lfs.github.com/ in order to push/pull large files on GitHub.

If you are working on a new environment, suggest running the following files to ensure all required libraries and packages to run the program are included:

> Found in `./code/`:
> - `requirements.txt`: Stores the required dependencies to be installed in a virtual environment. Run `pip install -r requirements.txt` in your shell.
> - `setup.py`: After installing the dependencies, run this to download the packages required. Run `python setup.py` in your shell.

Other requirements include working knowledge of `Celery` and `Redis` and their installation on your local machine.
> - To activate a Celery worker, run `celery worker -A app.celery --loglevel=info` on a new terminal (with also the virtual environment which Celery is installed on activated). Sometimes, the above command may not work for your machine. Try, instead, `celery worker -A app.celery -P solo --loglevel=info`.
> - To activate Redis, run `redis-server` on a separate new terminal.

## Dataset

This project will examine recipe data from Food.com – https://www.food.com.

Further, data has already been crawled from said website and available on Kaggle – https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions. The dataset has over 2000,000 recipes covering eighteen years of uploads on Food.com up to 2019.

Files used are `RAW_recipes.csv` and `RAW_interactions.csv`. To run these files in tandem with the code, download them from Kaggle and save them in the folder `./data/`.

## Recipe Recommendations

Topic Modeling will be performed on the dataset in order to extract the latent topics occurring in the large collection of recipes. Following which we will use these topics to separate the recipes into different groups/topics. The choice of topic modeling algorithm will be Latent Dirichlet Allocation (LDA).

### Code

Found in `./code/getRecommendations/`:

1. `topic_model.ipynb`: Notebook that performs topic modeling using LDA algorithm on the dataset. Saves to disk the following:
> - `dtm`: Document Term Matrix. A vector representation of the documents.
> - `lda`: The LDA model.
> - `pyLDAvis`: A neat and handy visualisation of the topic modeling. Saved as html.

2. `topic_assignment.ipynb`: After topic modeling has been performed and we have a saved `dtm` and `lda` model, use this notebook to get the topic distribution of the recipes and to assign each recipe their topic(s). Saves to disk an `assignment` CSV file.

3. `topic_topN.ipynb`: Once we have our topics, we want to look at the Top N words for each topic. We also want to expand these words by including their synonyms, hyponyms, hypernyms, meronyms, holonyms & entailments. Saves to disk a `topN` JSON file.

3. `getRecommendations_terminal.py`: A python script that asks the user to input their likes and dislikes and matches those to the closest recipes. Requires an `assignment` CSV file and a `topN` JSON file.
> - Note: Attached is an example screenshot of the terminal output of the program:
![2021-11-13 02_26_29-Command Prompt](https://user-images.githubusercontent.com/19281828/141614709-9cb1f244-d129-43b6-9356-def2cb82a257.png)

### Data

Found in `./data/getRecommendations/`:

(Note: experiment files that are not used in the final model are not included in the repo. Done for brevity sake and to reduce size of repo.)

Current experiment version used in model: `exp3`.
 
- `experiments.txt`: Various experiment designs are described here and their results recorded.
- `assignments/`: Stores each experiment's CSV file that contains the recipe-topic assignments.
- `dtm/`: Stores each experiment's saved DTM.
- `lda/`: Stores each experiment's saved LDA model.
- `pyLDAvis/`: Stores each experiment's visualisation html file.
- `stopwords/`: Stores each experiment's stopwords (if any). Used in the token pre-processing step in `topic_model.ipynb`.
- `topN/`: Stores each experiment's topics' expanded top N words.

## Find Similar Recipes

Document similarity will be performed between a target recipe and the dataset to find a list of closely related recipes from the dataset. The method Word Mover’s Distance (WMD) will be used to calculate the document similarity.

### Code

Found in `./code/getSimilar/`:

1. `document_similarity.ipynb`: Notebook that creates the WMD model trained on the word vectors from the dataset. Saves to disk the following:
> - `wmd.model`: The WMD model. The model has been defined to retrieve the top 5 results.

2. `getSimilar_terminal.py`: A python script that asks the user to input a foodnetwork.com recipe url, performs web scrapping on it and uses the WMD model to match the information to the closest related recipes from the dataset. Requires a `wmd.model` file and the dataset CSV file.
> - Note 1: Depending if foodnetwork.com changes or modifies their website, this program may or may not work thereafter. Works as of Sep 2021.
> - Note 2: The program takes extremely, EXTREMELY long to run (>2hrs). This is because using WMD and word vectors is not a very time efficient algorithm. Furthermore, the dataset being compared with is huge, hence, the long wait. However, results are still satisfactory. Future research can be done to identify another method (neural networks perhaps?).
> - Note 3: Due to the very impractical long time required to execute the program fully to obtain the results, this program will only be used as an experimental proof and will not be included in the final product. 
> - Note 4: Attached is an example screenshot of the terminal output of the program:
![2021-09-24 21_49_55-Window](https://user-images.githubusercontent.com/19281828/134685571-b73ad552-98f0-463a-a572-8b777918eeb3.png)

3. `similarity_evaluation.ipynb`: Notebook that measures the Word Mover's Distance value between a sample target recipe with the model's output to evaluate their similarity.

### Data

Found in `./data/getSimilar/`:

- `wmd.model`: The WMD model trained on the word vectors from the dataset. The model has been defined to retrieve the top 5 results.

## Tags Generator

A seq2seq transformer that takes in the text data of a recipe's steps and outputs tags for the recipe.

### Code

Found in `./code/getTags/`:

1. `build_transformer.ipynb`: Notebook that creates the transformer and trains it on the dataset. Current built: Google Colab GPU trained on 10,000 rows from dataset. Saves to disk the following:
> - `inp_tokenizer.pickle`: Input tokenizer, built from 'steps' column of dataset.
> - `input_tensor.pickle`: Input tensor, built from 'steps' column of dataset.
> - `targ_tokenizer.pickle`: Target tokenizer, built from 'tags' column of dataset.
> - `target_tensor.pickle`: Target tensor, built from 'tags' column of dataset.
> - `transformer.index` + `transformer.data` + `checkpoint`: Saved weights of transformer, based on current built.

2. `run_transformer.py`: A python script that loads the transformer, runs it, generates the tags and returns said tags. Requires input tensor, input tokenizer, target tensor and target tokenizer. Also transformer model weights.

3. `getTags_terminal.py`: A python script that asks the user to input a foodnetwork.com recipe url, performs web scrapping on it and inputs the data into the transformer to return tags for the recipe. Requires run_transformer.py.
> - Note: Attached is an example screenshot of the terminal output of the program:
![2021-10-05 17_55_05-Command Prompt](https://user-images.githubusercontent.com/19281828/136001743-bebb2700-bad4-4aa6-b279-4d982fba90ae.png)

### Data

Found in `./data/getTags/`:

- `inp_tokenizer.pickle`: Input tokenizer.
- `input_tensor.pickle`: Input tensor.
- `targ_tokenizer.pickle`: Target tokenizer.
- `target_tensor.pickle`: Target tensor.
- `transformer.index` + `transformer.data` + `checkpoint`: Saved weights of transformer.

## Ratings Classification

Using Hugging Face 🤗 API and taking advantage of transfer learning, a BERT model (more specifically the lightweight DistilBERT base model) was fine tuned with a custom dataset from RAW_interactions.csv to perform multi-label classification. The classification would be the rating values of 1-5.

A user inputed review will be passed into the above to automatically classify it with a rating.

### Code

Found in `./code/getRatings/`:

1. `fine_tune_BERT.ipynb`: This notebook takes a custom dataset - 'RAW_interactions.csv' and preprocess the reviews and ratings columns to be usable with DistilBERT base model from Hugging Face. Then fine-tune said model according to the custom dataset to perform multi-labels classification (the ratings from 1-5). Saves to disk the following:
> - `config.json` and `tf_model.h5`: The weights of the fine-tuned model.

2. `run_BERT.py`: A python script that takes in a review and runs it through the fine-tuned DistilBERT base model from Huggingface. The results from the model is then post-processed to get the rating number.

3. `getRatings_terminal.py`: A python script that asks the user to input a review for a recipe. The script will then automatically give a rating (from 1-5) of the recipe based on the review. Requires run_BERT.py
> - Note: Attached are example screenshots of the terminal output of the program:
![2021-11-07 15_08_52-Command Prompt](https://user-images.githubusercontent.com/19281828/140636069-c3c81882-5c9c-4c64-b7e1-5eb2f7aa06b2.png)
![2021-11-07 15_29_06-Command Prompt](https://user-images.githubusercontent.com/19281828/140636409-4254744e-7408-4220-9f38-2a758b88b9e1.png)
![2021-11-07 15_14_52-Command Prompt](https://user-images.githubusercontent.com/19281828/140636073-50c13614-a670-4235-a4f3-0609fef4a6eb.png)
![2021-11-07 15_26_49-Command Prompt](https://user-images.githubusercontent.com/19281828/140636364-aeeb0b79-f98e-4ce3-8751-ce44e6d639b3.png)
![2021-11-07 15_27_31-Command Prompt](https://user-images.githubusercontent.com/19281828/140636366-c2685a56-62c4-4742-a1b0-64eccbde4c2b.png)

### Data

Found in `./data/getRatings/`:

- `config.json` and `tf_model.h5`: The weights of the fine-tuned model.

## Conversational UI

The CUI portion of the system will utilise the top-notch NLP solution provided by Google Dialogflow ES. It will be integrated with Telegram’s bots to create a chatbot on said platform. The name of the bot is `marcusyfoodbot`.

### Code

Found in `./code/cui/`:

1. `app.py`: The flask app that runs the server-side code that allows for robust handling of intents via fulfillment and webhook.
2. `config.py`: The configuration file for the flask app.
3. `getRecommendations_chatbot.py`: Similar to `getRecommendations_terminal.py` but with implementations specific for chatbot.
4. `getTags_chatbot.py`: Similar to `getTags_terminal.py` but with implementations specific for chatbot.
5. `getRatings_chatbot.py`: Similar to `getRatings_terminal.py` but with implementations specific for chatbot.
6. `agent.zip`: The Dialogflow agent.