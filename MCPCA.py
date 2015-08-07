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
	#print df.head()
	return df

def perform_PCA(df):
	threshold = 0.1
	pca = decomposition.PCA(n_components=3)
	numpyMatrix = df.as_matrix().astype(float)
	scaled_data = preprocessing.scale(numpyMatrix)
	pca.fit(scaled_data)	
	transformed=pca.transform(scaled_data)
	CompScores= pd.DataFrame(data=transformed,columns = ['PC1','PC2','PC3'],index=df.index)
	#print CompScores
	CompScores.to_csv('CompScores.csv')
	pca_components_df = pd.DataFrame(data = pca.components_,columns = df.columns.values)
	#print pca_components_df.T
	pca_components_df=pca_components_df.T
	#pca_components_df.to_csv('pca_components_df.csv')
	filtered = pca_components_df[abs(pca_components_df) > threshold]
	#print filtered
	#filtered.to_csv('filtered.csv')
	#print pca.explained_variance_ratio_
df = prepare_data(filename)
perform_PCA(df)
