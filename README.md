# Movie Data-Science Project

A CMPT353 project analyzing success in movies using data from Wikidata, Omdb, and Rotten Tomatoes.

## Required Libraries

import pandas as pd 
import numpy as np 
sklearn : {
sklearn.model_selection
sklearn.naive_bayes
sklearn.neighbors
sklearn.preprocessing
sklearn.pipeline import make_pipeline
sklearn.svm import SVC
sklearn.linear_model
}
matplotlib
seaborn
scipy
statsmodels.stats.multicomp


## How to run
All code used to produce the report is located notebooks in the main directory.
required: http://jupyter.org/install
to run:
`jupyter notebook`
### Basic Correlations.ipynb
Calculates correlation between audience reviews with profit and number of ratings with profit
### Director_vs_Actor_on_nBox.ipynb
Tests if directors/actors have impact on box office
### Genre_Profit_Notebook.ipynb
Tests if genre and profit are related
### NLP.ipynb
Predicts genre based on plot
### time_genre.ipynb
Shows what genres are popular during a certain year
### predict_user_ratings.ipynb
Predicts an Rotten Tomato audience rating

## How to run Data Extraction
https://coursys.sfu.ca/2018su-cmpt-353-d1/pages/ProjectMoviesData
