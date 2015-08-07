import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import decomposition, preprocessing


# data from www.gapminder.org/data/
filename = 'PCACG2.csv'

def prepare_data(filename):
	df=pd.read_csv(filename,index_col=0)
	df = df.fillna('0')
	return df

def perform_PCA(df):
	threshold = 0.1
	component = 1 #Second of two right now
	pca = decomposition.PCA(n_components=5)
	numpyMatrix = df.as_matrix().astype(float)
	scaled_data = preprocessing.scale(numpyMatrix)
	pca.fit(scaled_data)	
	pca.transform(scaled_data)
	
	pca_components_df = pd.DataFrame(data = pca.components_,columns = df.columns.values)
	#print pca_components_df
	#pca_components_df.to_csv('pca_components_df.csv')

	filtered = pca_components_df[abs(pca_components_df) > threshold]
	trans_filtered= filtered.T
	#print filtered.T #Tranformed Dataframe
	trans_filtered.to_csv('trans_filtered.csv')
	print pca.explained_variance_ratio_
df = prepare_data(filename)
perform_PCA(df)
