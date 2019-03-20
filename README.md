# Movie Success Analysis Project

Movie Success Analysis Project cooperating machine learning techniques and data sets from Wikidata, Omdb, and Rotten Tomatoes to analyze success in movies. The project is written in Python.

## Installation
Please visit the notebooks in the main directory for code used to produce the report. <br />
required: http://jupyter.org/install <br />
to run:
`jupyter notebook`

### Required Libraries

import pandas as pd <br />
import numpy as np <br />
sklearn : {<br />
sklearn.model_selection <br />
sklearn.naive_bayes <br />
sklearn.neighbors <br />
sklearn.preprocessing <br />
sklearn.pipeline import make_pipeline <br />
sklearn.svm import SVC <br />
sklearn.linear_model <br />
} <br />
matplotlib <br />
seaborn <br />
scipy <br />
statsmodels.stats.multicomp <br />

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


