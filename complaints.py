import os
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def preprocess():
    # loading data
    df = pd.read_csv('datasets/complaints.csv', low_memory=False)
    print(df.shape)
    print(df.head(3).T)
    # Create a new dataframe with two columns
    df1 = df[['Product', 'Consumer complaint narrative']].copy()
    # Remove missing values (NaN)
    df1 = df1[pd.notnull(df1['Consumer complaint narrative'])]
    # Renaming second column for a simpler name
    # print(df.columns)
    df1.columns = ['Product', 'Consumer_complaint']
    print(df1.shape)
    quit()
    # print(df1.head(3).T)
    df1_values = pd.DataFrame(df1.Product.unique()).values
    print(df1_values)
    # Because the computation is time consuming (in terms of CPU), the data was sampled
    df2 = df1.sample(10000, random_state=1).copy()
    # Renaming categories
    df2.replace({'Product':
                     {'Credit reporting': 'Credit reporting, repair, or other',
                      'Credit card': 'Credit card or prepaid card',
                      'Prepaid card': 'Credit card or prepaid card',
                      'Debt or credit management': 'Debt collection,credit management',
                      'Payday loan': 'Payday loan, title loan, or personal loan',
                      'Bank account': 'Bank account,checking and saving account or service',
                      'Money transfer': 'Money transfer, virtual currency, or money service',
                      'Consumer loan': 'Consumer loan, student loan,mortgage or vehicle loan',
                      'Other financial service': 'Other financial service',
                      'Virtual currency': 'Money transfer, virtual currency, or money service'
                      }
                 },
                inplace=True)
    df2_reduced_values = pd.DataFrame(df2.Product.unique())
    # Create a new column 'category_id' with encoded categories
    df2['category_id'] = df2['Product'].factorize()[0]
    category_id_df = df2[['Product', 'category_id']].drop_duplicates()
    # Dictionaries for future use
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    # New dataframe
    df2_new_head = df2.head()
    print(id_to_category)
    print(category_to_id)
    fig = plt.figure(figsize=(8, 6))
    colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey',
              'darkblue', 'darkblue', 'darkblue','darkblue']
    df2.groupby('Product').Consumer_complaint.count().sort_values().plot.barh(
        ylim=0, color=colors, title='NUMBER OF COMPLAINTS IN EACH PRODUCT CATEGORY')
    plt.xlabel('Number of ocurrences', fontsize=10)
    plt.show()

    # Transform texts into vectors using TFIDF and evaluate how important a particular word is in the collection of words.
    # Word importance is determined in terms of frequency. min_df:remove word occurred less than specified value
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2),
                            stop_words='english')
    # We transform each complaint into a vector
    features = tfidf.fit_transform(df2.Consumer_complaint).toarray()
    labels = df2.category_id
    print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" % (
        features.shape))
    # Finding the three most correlated terms with each of the product categories
    N = 3
    # for Product, category_id in sorted(category_to_id.items()):
    #     features_chi2 = chi2(features, labels == category_id)
    #     indices = np.argsort(features_chi2[0])
    #     feature_names = np.array(tfidf.get_feature_names_out())[indices]
    #     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    #     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #     print("n==> %s:" % (Product))
    #     print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
    #     print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))
if __name__=='__main__':
    preprocess()
