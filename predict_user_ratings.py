import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression



omdb = pd.read_json('movies/data/omdb-data.json.gz', orient='record', lines=True)
genre = pd.read_json('movies/data/genres.json.gz', orient='record', lines=True)
wiki = pd.read_json('movies/data/wikidata-movies2.json.gz', orient='record', lines=True)
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

def get_wins(str):
	token = str.split('.')
	for x in token:
		if x.find('wins') > -1:
			tmp = x.split(' ')
			return int(tmp[tmp.index('wins') -1])
		if x.find('win') > -1 :
			tmp = x.split(' ')
			return int(tmp[tmp.index('win') -1])
	return 0

def get_nominations(str):
	token = str.split('.')
	for x in token:
		if x.find('nominations') > -1:
			tmp = x.split(' ')
			return int(tmp[tmp.index('nominations') -1])
		if x.find('nomination') > -1 :
			tmp = x.split(' ')
			return int(tmp[tmp.index('nomination') -1])
	return 0


get_oscar_wins_vec = np.vectorize(get_oscar_wins)
get_oscar_nominations_vec = np.vectorize(get_oscar_nominations)
get_wins_vec = np.vectorize(get_wins)
get_nominations_vec = np.vectorize(get_nominations)


omdb['oscar_wins'] = get_oscar_wins_vec(omdb['omdb_awards'])
omdb['oscar_nominations'] = get_oscar_nominations_vec(omdb['omdb_awards'])
omdb['wins'] = get_wins_vec(omdb['omdb_awards'])
omdb['nominations'] = get_nominations_vec(omdb['omdb_awards'])



wiki_rotten = pd.merge(wiki, rotten,  how='left', left_on=['imdb_id','rotten_tomatoes_id'], right_on = ['imdb_id','rotten_tomatoes_id'])
omdb_wiki_rotten = pd.merge(wiki_rotten, omdb,  how='left', left_on=['imdb_id'], right_on = ['imdb_id'])

genres = omdb_wiki_rotten['omdb_genres'].str.join('|').str.get_dummies()
genres.reset_index(drop=True, inplace=True)
omdb_wiki_rotten = pd.concat([omdb_wiki_rotten, genres], axis=1)


del omdb_wiki_rotten['imdb_id']
del omdb_wiki_rotten['rotten_tomatoes_id']
del omdb_wiki_rotten['original_language']
del omdb_wiki_rotten['publication_date']
del omdb_wiki_rotten['series']
del omdb_wiki_rotten['wikidata_id']
del omdb_wiki_rotten['label']
del omdb_wiki_rotten['genre']
del omdb_wiki_rotten['filming_location']
del omdb_wiki_rotten['director']
del omdb_wiki_rotten['cast_member']
del omdb_wiki_rotten['based_on']
del omdb_wiki_rotten['country_of_origin']
del omdb_wiki_rotten['enwiki_title']
del omdb_wiki_rotten['metacritic_id']
del omdb_wiki_rotten['main_subject']
del omdb_wiki_rotten['omdb_genres']
del omdb_wiki_rotten['omdb_awards']
del omdb_wiki_rotten['omdb_plot']
# del omdb_wiki_rotten['critic_average']
# del omdb_wiki_rotten['critic_percent']
del omdb_wiki_rotten['audience_percent']


# print(omdb_wiki_rotten.columns.values)
omdb_wiki_rotten = omdb_wiki_rotten.dropna()

# omdb_wiki_rotten = omdb_wiki_rotten[['audience_average', 'audience_ratings', 'critic_average', 'critic_percent', 'nbox', 'ncost', 'wins', 'nominations']].dropna()
# X = omdb_wiki_rotten[['audience_ratings', 'critic_average', 'critic_percent', 'nbox', 'ncost', 'wins', 'nominations']]

X = omdb_wiki_rotten.drop(['audience_average'], axis=1)
y = omdb_wiki_rotten['audience_average']
y = y.apply(lambda x: round(x * 2) / 2) 
y = y.astype('str')
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# model = GaussianNB()
# model = KNeighborsClassifier(n_neighbors=20)
# model = SVC(kernel='linear', C=1.0)

model = make_pipeline(
    StandardScaler(),
    # KNeighborsClassifier(n_neighbors=10)
    # LinearRegression()
    SVC(kernel='linear', C=1)

)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# wiki['genre'] = wiki['genre'].apply(lambda x: x[0])
# wiki_genre = pd.merge(wiki, genre,  how='left', left_on=['genre'], right_on = ['wikidata_id']).dropna()
# print(wiki_genre)





# OMDB-DATA 
# {"imdb_id","omdb_genres" ,"omdb_plot","omdb_awards"}

# ROTTEN-TOMATOES
#{"audience_average","audience_percent","audience_ratings","critic_average","critic_percent","imdb_id":,"rotten_tomatoes_id"}

# WIKIDATA-MOVIES
# {"wikidata_id","label","imdb_id","rotten_tomatoes_id","genre","director":["Q43079418"],"cast_member":["Q228931","Q235384"],"publication_date":"2012-01-01","country_of_origin":"Q145","original_language":"Q1860"}

# Genres
# {"wikidata_id":"Q43334491","genre_label":"novella"}



