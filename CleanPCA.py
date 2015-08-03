import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import decomposition, preprocessing
#for interactive plotting with hover
import mpld3


# data from www.gapminder.org/data/
filenames = ["completion_male.csv", "completion_female.csv", "income_per_person.csv",
             "employment_over_15.csv", "life_expectancy.csv"]
             
gapminder_data = []
def prepare_data(filenames):
	for name in filenames: gapminder_data.append(pd.read_csv(name, index_col=0))
	# create a dataframe with multiple features per country
	df = pd.concat(gapminder_data, join="inner", axis=1)
	df.dropna(axis=0,how='any',inplace=True)
	# take the log of income_per_person
	# - this variables follows approximately a log-normal distribution bc it's money
	df.income_per_person = np.log(df.income_per_person)	
	#print df.head()    		         
	#print df.shape
	#df.to_csv('Test.csv')
	return df

def perform_PCA(df):
	pca = decomposition.PCA(n_components=2)
	scaled_data = preprocessing.scale(df)
	pca.fit(scaled_data)
	transformed = pca.transform(scaled_data)
	print pca.components_
	print pca.explained_variance_ratio_

	
df = prepare_data(filenames)
perform_PCA(df)

#Factor Analysis
def perform_FA(df):
	data_normal = preprocessing.scale(data) #Normalization
	fa = decomposition.FactorAnalysis(n_components=1)
	fa.fit(data_normal)
	print fa.components_ # Factor loadings