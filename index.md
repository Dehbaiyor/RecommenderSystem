# Import


```python
import re
import numpy as np
import scipy
import math
import random
import sklearn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.pipeline import Pipeline
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) 
from sklearn.ensemble import GradientBoostingClassifier as GBC
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    

# Data Loading and Preprocessing


```python
articles_df = pd.read_csv('articles_info.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.shape
```




    (2786, 5)




```python
articles_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>timestamp</th>
      <th>eventType</th>
      <th>contentId</th>
      <th>authorPersonId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A victim of the anti-police shooting in Baton ...</td>
      <td>1459193988</td>
      <td>CONTENT SHARED</td>
      <td>8712996026808754180</td>
      <td>4.340000e+18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Natural disasters like Hurricane Irma prompt p...</td>
      <td>1459194146</td>
      <td>CONTENT SHARED</td>
      <td>5431399346444238856</td>
      <td>4.340000e+18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York Times movie critic Andy Webster made ...</td>
      <td>1459194474</td>
      <td>CONTENT SHARED</td>
      <td>-1662763254714212341</td>
      <td>3.890000e+18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Former Department of Homeland Security (DHS) S...</td>
      <td>1459194497</td>
      <td>CONTENT SHARED</td>
      <td>381577463519666187</td>
      <td>4.340000e+18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Pentagon has reportedly excessively charge...</td>
      <td>1459194522</td>
      <td>CONTENT SHARED</td>
      <td>-8282357075607822317</td>
      <td>4.340000e+18</td>
    </tr>
  </tbody>
</table>
</div>




```python
interactions_df = pd.read_csv('users_interactions.csv')
interactions_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>eventType</th>
      <th>contentId</th>
      <th>personId</th>
      <th>sessionId</th>
      <th>userAgent</th>
      <th>userRegion</th>
      <th>userCountry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1465413032</td>
      <td>VIEW</td>
      <td>-3499919498720038879</td>
      <td>-8845298781299428018</td>
      <td>1264196770339959068</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1465412560</td>
      <td>VIEW</td>
      <td>8890720798209849691</td>
      <td>-1032019229384696495</td>
      <td>3621737643587579081</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2...</td>
      <td>NY</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1465416190</td>
      <td>VIEW</td>
      <td>310515487419366995</td>
      <td>-1130272294246983140</td>
      <td>2631864456530402479</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1465413895</td>
      <td>FOLLOW</td>
      <td>310515487419366995</td>
      <td>344280948527967603</td>
      <td>-3167637573980064150</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465412290</td>
      <td>VIEW</td>
      <td>-7820640624231356730</td>
      <td>-445337111692715325</td>
      <td>5611481178424124714</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Converting implicit user preferences to a rating scheme for recommender system


```python
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])
```


```python
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))
```

    # users: 1895
    # users with at least 5 interactions: 1140
    


```python
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
```

    # of interactions: 72312
    # of interactions from users with at least 5 interactions: 69868
    


```python
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(5)
```

    # of unique user/item interactions: 39106
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>personId</th>
      <th>contentId</th>
      <th>eventStrength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-9223121837663643404</td>
      <td>-8949113594875411859</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-9223121837663643404</td>
      <td>-8377626164558006982</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-9223121837663643404</td>
      <td>-8208801367848627943</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-9223121837663643404</td>
      <td>-8187220755213888616</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-9223121837663643404</td>
      <td>-7423191370472335463</td>
      <td>3.169925</td>
    </tr>
  </tbody>
</table>
</div>




```python
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
```

    # interactions on Train set: 31284
    # interactions on Test set: 7822
    


```python
#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')
```


```python
def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
```

## Building An Evaluator Class for the Recommender System using Holdout Validation


```python
#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 30

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (n) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the n random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of n non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': np.round(global_recall_at_5,2) * 100,
                          'recall@10': np.round(global_recall_at_10, 2) * 100}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()    
```

# Training and Evaluating the Recommendation Algorithm


```python
#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>contentId</th>
      <th>-9222795471790223670</th>
      <th>-9216926795620865886</th>
      <th>-9194572880052200111</th>
      <th>-9192549002213406534</th>
      <th>-9190737901804729417</th>
      <th>-9189659052158407108</th>
      <th>-9176143510534135851</th>
      <th>-9172673334835262304</th>
      <th>-9171475473795142532</th>
      <th>-9166778629773133902</th>
      <th>...</th>
      <th>9191014301634017491</th>
      <th>9207286802575546269</th>
      <th>9208127165664287660</th>
      <th>9209629151177723638</th>
      <th>9209886322932807692</th>
      <th>9213260650272029784</th>
      <th>9215261273565326920</th>
      <th>9217155070834564627</th>
      <th>9220445660318725468</th>
      <th>9222265156747237864</th>
    </tr>
    <tr>
      <th>personId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-9223121837663643404</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-9212075797126931087</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-9207251133131336884</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-9199575329909162940</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-9196668942822132778</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2926 columns</p>
</div>




```python
users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
users_ids = list(users_items_pivot_matrix_df.index)
users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
```


```python
#The number of factors to factor the user-item matrix. This is the tuning parameter 
NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
#U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
```


```python
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
```


```python
all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
```


```python
#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-9223121837663643404</th>
      <th>-9212075797126931087</th>
      <th>-9207251133131336884</th>
      <th>-9199575329909162940</th>
      <th>-9196668942822132778</th>
      <th>-9188188261933657343</th>
      <th>-9172914609055320039</th>
      <th>-9156344805277471150</th>
      <th>-9120685872592674274</th>
      <th>-9109785559521267180</th>
      <th>...</th>
      <th>9105269044962898535</th>
      <th>9109075639526981934</th>
      <th>9135582630122950040</th>
      <th>9137372837662939523</th>
      <th>9148269800512008413</th>
      <th>9165571805999894845</th>
      <th>9187866633451383747</th>
      <th>9191849144618614467</th>
      <th>9199170757466086545</th>
      <th>9210530975708218054</th>
    </tr>
    <tr>
      <th>contentId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-9222795471790223670</th>
      <td>0.138279</td>
      <td>0.137371</td>
      <td>0.136094</td>
      <td>0.143370</td>
      <td>0.136233</td>
      <td>0.136697</td>
      <td>0.136831</td>
      <td>0.143052</td>
      <td>0.136263</td>
      <td>0.135138</td>
      <td>...</td>
      <td>0.136478</td>
      <td>0.129214</td>
      <td>0.137378</td>
      <td>0.138938</td>
      <td>0.139506</td>
      <td>0.136401</td>
      <td>0.135032</td>
      <td>0.134333</td>
      <td>0.135138</td>
      <td>0.136155</td>
    </tr>
    <tr>
      <th>-9216926795620865886</th>
      <td>0.137479</td>
      <td>0.137359</td>
      <td>0.138125</td>
      <td>0.137334</td>
      <td>0.137415</td>
      <td>0.137432</td>
      <td>0.137417</td>
      <td>0.137501</td>
      <td>0.137674</td>
      <td>0.137577</td>
      <td>...</td>
      <td>0.137404</td>
      <td>0.138981</td>
      <td>0.137456</td>
      <td>0.137561</td>
      <td>0.139106</td>
      <td>0.137510</td>
      <td>0.137665</td>
      <td>0.138292</td>
      <td>0.137903</td>
      <td>0.137806</td>
    </tr>
    <tr>
      <th>-9194572880052200111</th>
      <td>0.135555</td>
      <td>0.137089</td>
      <td>0.136667</td>
      <td>0.136978</td>
      <td>0.139808</td>
      <td>0.137239</td>
      <td>0.140664</td>
      <td>0.135638</td>
      <td>0.134570</td>
      <td>0.138063</td>
      <td>...</td>
      <td>0.138687</td>
      <td>0.142793</td>
      <td>0.138535</td>
      <td>0.139449</td>
      <td>0.154870</td>
      <td>0.139662</td>
      <td>0.140099</td>
      <td>0.135138</td>
      <td>0.138903</td>
      <td>0.153150</td>
    </tr>
    <tr>
      <th>-9192549002213406534</th>
      <td>0.141372</td>
      <td>0.137412</td>
      <td>0.134219</td>
      <td>0.136539</td>
      <td>0.139294</td>
      <td>0.138089</td>
      <td>0.138757</td>
      <td>0.144309</td>
      <td>0.142802</td>
      <td>0.137761</td>
      <td>...</td>
      <td>0.139616</td>
      <td>0.165026</td>
      <td>0.138252</td>
      <td>0.136549</td>
      <td>0.141647</td>
      <td>0.138735</td>
      <td>0.138964</td>
      <td>0.136516</td>
      <td>0.140238</td>
      <td>0.147781</td>
    </tr>
    <tr>
      <th>-9190737901804729417</th>
      <td>0.139581</td>
      <td>0.136905</td>
      <td>0.138139</td>
      <td>0.138051</td>
      <td>0.137206</td>
      <td>0.137655</td>
      <td>0.137856</td>
      <td>0.137837</td>
      <td>0.134453</td>
      <td>0.139556</td>
      <td>...</td>
      <td>0.137790</td>
      <td>0.137684</td>
      <td>0.137579</td>
      <td>0.137428</td>
      <td>0.133775</td>
      <td>0.137301</td>
      <td>0.137525</td>
      <td>0.137632</td>
      <td>0.138044</td>
      <td>0.135815</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1140 columns</p>
</div>




```python
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'right', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['contentId', 'text', 'recStrength']]


        return recommendations_df.dropna().head(topn*3)
```


```python
recsys = CFRecommender(cf_preds_df, articles_df)

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(recsys)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
```

    Evaluating Collaborative Filtering (SVD Matrix Factorization) model...
    1139 users processed
    
    Global metrics:
    {'modelName': 'Collaborative Filtering', 'recall@5': 56.00000000000001, 'recall@10': 71.0}
    


```python
cf_detailed_results_df.sort_values('interacted_count',ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hits@5_count</th>
      <th>hits@10_count</th>
      <th>interacted_count</th>
      <th>recall@5</th>
      <th>recall@10</th>
      <th>_person_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76</th>
      <td>64</td>
      <td>100</td>
      <td>192</td>
      <td>0.333333</td>
      <td>0.520833</td>
      <td>3609194402293569455</td>
    </tr>
    <tr>
      <th>17</th>
      <td>64</td>
      <td>95</td>
      <td>134</td>
      <td>0.477612</td>
      <td>0.708955</td>
      <td>-2626634673110551643</td>
    </tr>
    <tr>
      <th>16</th>
      <td>43</td>
      <td>59</td>
      <td>130</td>
      <td>0.330769</td>
      <td>0.453846</td>
      <td>-1032019229384696495</td>
    </tr>
    <tr>
      <th>10</th>
      <td>65</td>
      <td>78</td>
      <td>117</td>
      <td>0.555556</td>
      <td>0.666667</td>
      <td>-1443636648652872475</td>
    </tr>
    <tr>
      <th>82</th>
      <td>58</td>
      <td>70</td>
      <td>88</td>
      <td>0.659091</td>
      <td>0.795455</td>
      <td>-2979881261169775358</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-4716637009380371036</td>
    </tr>
    <tr>
      <th>1037</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7371331529308244882</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-2108111265827636778</td>
    </tr>
    <tr>
      <th>1110</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3011221777935346594</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1306495113310433590</td>
    </tr>
  </tbody>
</table>
<p>1140 rows × 6 columns</p>
</div>



## Saving the trained Recommender for later use


```python
#pickle.dump(cf_preds_df, open('recmat_1.cf', 'wb'))
#pickle.dump(articles_df, open('recmat_2.cf', 'wb'))
```


```python

```
