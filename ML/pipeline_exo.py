import pandas as pd

# import data
# enter your local path to the data
path_data_train = ""
train = pd.DataFrame.from_csv(
    path_data_train, encoding="ISO-8859-1", index_col=None)
path_data_test = ""
test = pd.DataFrame.from_csv(
    path_data_test, encoding="ISO-8859-1", index_col=None)

# extract data
texts_train = train['tweet'].fillna("").values
labels_train = train['sentiment'].values

texts_test = train['tweet'].fillna("").values
labels_test = train['sentiment'].values

# seed to be used to replicate identical results
SEED = 42

""" preprocess the input data """
""" trasnform the data using the tf-idf """
""" can be done in two or one step """

# preprocessing (not limited at)
import re
from nltk.stem import SnowballStemmer
from nltk import RegexpTokenizer

# tf-idf transformation
from sklearn.feature_extraction.text import TfidfVectorizer

""" perform a logistic regression """
""" choose the optimal regularization coefficient C using a 3-fold CV """
""" predict the test data using the best model """
""" diagnostic the results """
""" cross validation can be done manually or using in-built function """

# logistic regression
from sklearn.linear_model import LogisticRegression
# cross validation folds generator
from sklearn.model_selection import KFold
# automatic grid search
from sklearn.model_selection import GridSearchCV

# diagnostic tools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

""" perform a multinomial NB """
""" choose the optimal alpha """
""" predict the results """
""" diagnostic the results """
""" choose the best model """

# can be used as transformation instead of the tf-idf (normal count)
from sklearn.feature_extraction.text import CountVectorizer
# multinomial NB
from sklearn.naive_bayes import MultinomialNB

""" build a pipeline with the best model """
""" use CV to find the best parameter at each step of the pipeline """
# make a pipeline
from sklearn.pipeline import Pipeline
# can be use with countVectorizez to perform a tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
