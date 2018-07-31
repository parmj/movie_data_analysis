import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC






omdb = pd.read_json('movies/data/omdb-data.json.gz', orient='record', lines=True)
genre = pd.read_json('movies/data/genres.json.gz', orient='record', lines=True)
wiki = pd.read_json('movies/data/wikidata-movies.json.gz', orient='record', lines=True)
rotten = pd.read_json('movies/data/rotten-tomatoes.json.gz', orient='record', lines=True)

# data = pd.read_json('label_map/part-00000-2ab6072d-bb8d-475a-b4c2-190e48e5befe-c000.json', lines=True)

test_string = 'Nominated for 2 Oscars. Another 2 nominations.'

def get_oscar_wins(str):
	token = str.split('.')
	if token[0].find('Won') > -1:
		return int(token[0].split(' ')[1])
	else :
		return 0

def get_oscar_nominations(str):
	token = str.split('.')
	if (token[0].find('Nominated') > -1) & (token[0].find('Oscar') > -1):
		return int(token[0].split(' ')[2])
	else :
		return 0

get_oscar_wins_vec = np.vectorize(get_oscar_wins)
get_oscar_nominations_vec = np.vectorize(get_oscar_nominations)


omdb['oscar_wins'] = get_oscar_wins_vec(omdb['omdb_awards'])
omdb['oscar_nominations'] = get_oscar_nominations_vec(omdb['omdb_awards'])

# omdb_genre = omdb.join(genre.set_index('wikidata_id'), )

# wiki_rotten = pd.merge(wiki, rotten,  how='left', left_on=['imdb_id','rotten_tomatoes_id'], right_on = ['imdb_id','rotten_tomatoes_id'])
# X = wiki_rotten.drop(['audience_percent', 'audience_ratings','audience_average', 'imdb_id', 'rotten_tomatoes_id', 'wikidata_id'], axis=1)

omdb_rotten = pd.merge(omdb, rotten,  how='left', left_on=['imdb_id'], right_on = ['imdb_id']).dropna()

X = omdb_rotten[['audience_ratings', 'critic_average', 'critic_percent', 'oscar_wins', 'oscar_nominations']]
y = omdb_rotten['audience_average'].values
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y)

# model = GaussianNB()
# model = KNeighborsClassifier(n_neighbors=20)
# model = SVC(kernel='linear', C=2.0)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=5)
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))







# OMDB-DATA 
# {"imdb_id","omdb_genres" ,"omdb_plot","omdb_awards"}

# ROTTEN-TOMATOES
#{"audience_average","audience_percent","audience_ratings","critic_average","critic_percent","imdb_id":,"rotten_tomatoes_id"}

# WIKIDATA-MOVIES
# {"wikidata_id","label","imdb_id","rotten_tomatoes_id","genre","director":["Q43079418"],"cast_member":["Q228931","Q235384"],"publication_date":"2012-01-01","country_of_origin":"Q145","original_language":"Q1860"}

# Genres
# {"wikidata_id":"Q43334491","genre_label":"novella"}



