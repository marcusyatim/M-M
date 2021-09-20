
# Mings & Marcus

 
Note: A lot of large files in data folder. Suggest to install `git lfs` @ https://git-lfs.github.com/ in order to push/pull large files on GitHub.

## Dataset
This project will examine recipe data from Food.com – https://www.food.com. 

Further, data has already been crawled from said website and available on Kaggle – https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions. The dataset has over 2000,000 recipes covering eighteen years of uploads on Food.com up to 2019.

Found in `./data/RAW_recipes.csv/`.

## Topic Model
Topic Modeling will be performed on the dataset in order to extract the latent topics occurring in the large collection of recipes. Following which we will use these topics to separate the recipes into different groups/topics. The choice of topic modeling algorithm will be Latent Dirichlet Allocation (LDA).

### Code
From `./code/topic-model/`:

 1. `topic_model.ipynb`: Notebook that performs topic modeling using LDA algorithm on the dataset. Saves to disk the following:

> - `tokens`: Word tokens formed after pre-processing and tokenisation of dataset.
> - `dict`: Use the gensim library's corpus function to create a dictionary of the word tokens and further filter them based on tf-idf (i.e. when a word appears in many documents, it’s considered unimportant. When the word is relatively unique and appears in few documents, it’s important).
> - `dtm`: Document Term Matrix. A vector representation of the documents.
> - `lda`: The LDA model.
> - `pyLDAvis`: A neat and handy visualisation of the topic modeling. Saved as html.

 2.  `topic_assignment.ipynb`: After topic modeling has been performed and we have a saved `model` and `dtm`, use this notebook to get the topic distribution of the recipes and to assign each recipe their topic(s). Saves to disk a new CSV file with the assignment.

### Data
From `./data/topic-model/`:

 - `experiments.txt`: Various experiment designs are described here and their results recorded.
 - `csvs/`: Stores each experiment's CSV file that contains the recipe-topic assignments.
 - `dicts/`: Stores each experiment's saved dictionary.
 - `dtms/`: Stores each experiment's saved DTMs.
 -  `models/`: Stores each experiment's saved LDA model.
 -  `pyLDAvis/`: Stores each experiment's visualisation html file.
 -  `stopwords/`: Stores each experiment's stopwords (if any). Used in the token pre-processing step in `topic_model.ipynb`.
 - `tokens/`: Stores each experiment's word tokens.