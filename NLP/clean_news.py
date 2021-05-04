#Abdul Sharopov and Abhaya Gauchan
# packages for our program
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import argparse
import re
re.compile('<title>(.*)</title>')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
def enable_parsing(parser):
    parser.add_argument("--train", help="Training data", type=str,
                        required=True)
    parser.add_argument("--test", help="Test data",type=str,
                        required=True)
    parser.add_argument("--pred", help="Output file",type=str, required=True)
    parser.add_argument("--max_feat", help="Max features used TfidfVectorizer",
                        type=int, required=True)
    parser.add_argument('--num_folds', help="Folds for CV",
                        required=True, type=int)
    return parser
parser = enable_parsing(parser)
args = parser.parse_args()
dir = "./"
TRAIN = args.train
TEST = args.test
PRED = args.pred
MAX_FEAT = args.max_feat
NUM_FOLDS = args.num_folds

#if __name__=='__main__':
input_train_data=pd.read_csv(TRAIN)
input_test_data=pd.read_csv(TEST)

# split the input dataframe into different dataframes with independent (x) and dependent (y) each [for simplicity of analysis for now]
# I do not think we actually need to split them, but it is an easy fix. later on we can just specify the independent and depent variables 
# of requiring two dataframes for the two variables

input_train_x=input_train_data[['Sentence']].copy()
input_train_y=input_train_data[['Bad Sentence']].copy()

input_test_x=input_test_data[['Sentence']].copy()
input_test_y=input_test_data[['Bad Sentence']].copy()

# Not entirely sure what these classes/functions are for, but I know that they feed into our pipelines in the next chunk
# it is explained in chapter two of our class textbook

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

#class Text_Preprocess(TextSelector):
  #def __init__(self, field, str_input):
    #super().__init__(field)
    #self.str_input=str_input
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str(str_input)).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words

# pipeline for the TF-IDF vectorizer
# user has to input max features as a parameter in the vectorizer
classifier = Pipeline([
                     ('colext', TextSelector('input_train_data')),
                     ('tfidf', TfidfVectorizer(tokenizer=Tokenizer,max_features= MAX_FEAT, stop_words='english',
                                               min_df=.0025, max_df=0.25, ngram_range=(1,3))),
                     ('svd', TruncatedSVD(algorithm='randomized', n_components=300)),
                     ('xgbclf', XGBClassifier( n_estimators=300,
                                               learning_rate=0.1, colsample_bytree=1, gamma =0.1,
                                              num_class=1, reg_lambda=7, objective='binary:logistic'))
                     ])

#Lists valid parameters for GridSearchCV
classifier.get_params().keys()

#define the parameters required for the text XGB classifier (test_pipe)
''''xgbclf__n_estimators':(50,100,200,300),
    'xgbclf__reg_lambda':(1,4,7),
    'xgbclf__num_class':(1,2,3,4),
    'xgbclf__gamma':(0.1,0.3,0.5),

parameters={
    'xgbclf__learning_rate':(0.0001, 0.01, 0.1, 0.5),
    'xgbclf__colsample_bytree':(0.3,0.7,1),
    'xgbclf__max_depth':(1,3,7),
    'xgbclf__objective':('multi:softmax','binary:logistic')
    }'''

from sklearn.metrics import classification_report
GridSearch=GridSearchCV(classifier, param_grid=parameters, scoring = 'f1', n_jobs=-1,verbose=3, cv=NUM_FOLDS)
GridSearch.fit(input_train_x,input_train_y.values.ravel())

print ('Best score: %0.3f' % GridSearch.best_score_)
print ('Best parameters set:')
best_parameters = GridSearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = GridSearch.predict(input_test_x)
print(classification_report(input_test_y, predictions))

# Fitting the model on our training data
classifier.fit(input_train_x, input_train_y)
# predicting labels based on test data
preds = classifier.predict(input_test_x)

#Output pred_data file
preds_df = input_test_data.drop('Bad Sentence', axis=1)
preds_df['Predicted Bad Sentence'] = pd.Series(preds, index=preds_df.index)
preds_df.to_csv(str(PRED))