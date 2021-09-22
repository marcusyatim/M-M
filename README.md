# Mings & Marcus

Note: Some large files in data folder. Suggest to install `git lfs` @ https://git-lfs.github.com/ in order to push/pull large files on GitHub.

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

2. `topic_assignment.ipynb`: After topic modeling has been performed and we have a saved `model` and `dtm`, use this notebook to get the topic distribution of the recipes and to assign each recipe their topic(s). Saves to disk an `assignment` CSV file.

3. `topic_topN.ipynb`: Once we have our topics, we want to look at the Top N words for each topic. We also want to expand these words by including their synonyms, hyponyms, hypernyms, meronyms, holonyms & entailments. Saves to disk a `topN` JSON file.

3. `topic_matching_terminal.py`: A python script that asks the user to input their likes and dislikes and matches those to the closest recipes. Requires an `assignment` CSV file and a `topN` JSON file.

`requirements.txt`: Stores the required dependencies to be installed in a virtual environment. Run `pip install -r requirements.txt` in your shell.
`setup.py`: After installing the dependencies, run this to download the packages required. Run `python setup.py` in your shell. 

### Data

From `./data/topic-model/`:

(Note: experiment files that are not used in the final model are not included in the repo. Done for brevity sake and to reduce size of repo.)

Current experiment version used in model: `exp3`.
 
- `experiments.txt`: Various experiment designs are described here and their results recorded.
- `assignments/`: Stores each experiment's CSV file that contains the recipe-topic assignments.
- `dicts/`: Stores each experiment's saved dictionary.
- `dtms/`: Stores each experiment's saved DTMs.
- `models/`: Stores each experiment's saved LDA model.
- `pyLDAvis/`: Stores each experiment's visualisation html file.
- `stopwords/`: Stores each experiment's stopwords (if any). Used in the token pre-processing step in `topic_model.ipynb`.
- `tokens/`: Stores each experiment's word tokens.
- `topN/`: Stores each experiment's topics' expanded top N words.