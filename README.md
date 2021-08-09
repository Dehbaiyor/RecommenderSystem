# News-Recommeder-System
This repo contains the code for an hybrid news recommender system that attempts to recommend news items along the user's preferential moral lines

# Recommender System
There were three different components of the recommender system:
- The hybrid recommender system.
- The boosting model for multi-label morality prediction
- The LSTM model for muti-label morality prediction

# The Hybrid Recommender System
This used both a collaborative filtering based recommender system and a content based recommender system.
## Colaborative Filtering System
The collaborative fitering system tries to measure users sililarity based on interactions such as viws, likes, comments etc.
The system then proceeds to use dot product calculations of the user-news matrix factorization to calculate similarity.
## Content Based System
The content based system using a tf-idf representation of all the news corpus alongside the morality prediction from both the boosting and LSTM models 
to find similarity between the items the user have interacted with and those they have not to make recommenders.

# The Boosting Model
The boosting model uses one-vs-rest classification (as this is a multilabel classification) with 100 estimators and 
a trigram tf-idf representation of the news corpus to build a multi-label morality prediction.

# The LSTM Model
The deep LSTM model uses an embedding size of 200 with 1024 hidden neuron in the LSTM layer and a loss function of 
binary-crossentropy to build a multi-label morality prediction.

# Usage
An weighted score from the content based and collaborative filtering system is used to make the final recommendations.
