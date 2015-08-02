#####
'''
Perform a test at the 1st CoGrp level. Then Drill down to deeper levels exluding more an more
items. Take the strongest items and see if i can derive anything meaningful from it.
'''
#####
'''
Although a PCA applied on binary data would yield results comparable to those obtained from a

Multiple Correspondence Analysis (factor scores and eigenvalues are linearly related), 

there are more appropriate techniques to deal with mixed data types, 
namely :

Multiple Factor Analysis

for mixed data available in the FactoMineR R package (AFDM()).
If your variables can be considered as structured subsets of descriptive attributes,
then Multiple Factor Analysis (MFA()) is also an option.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import decomposition, preprocessing

# for interactive plotting with hover
import mpld3
from mpld3 import plugins

# data from www.gapminder.org/data/
filenames = ["completion_male.csv", "completion_female.csv", "income_per_person.csv",
             "employment_over_15.csv", "life_expectancy.csv"]
             
gapminder_data = []
for name in filenames: gapminder_data.append(pd.read_csv(name, index_col=0))
# create a dataframe with multiple features per country

df = pd.concat(gapminder_data, join="inner", axis=1)
#print df.head(5) 
          
#What the above just did was join together a bunch of datafames on the index 'country'
# keep only countries for which we have a complete set of features
# (no missing data in any of the columns)

df.dropna(axis=0, how="any", inplace=True)
#if any na are present. Drop that label.
#So now we have complete rows for every record. All non complete rows were dropped.

# take the log of income_per_person 
# - this variable follows approximately a log-normal distribution
#Common for MONATARY VALUES(ex: LTV)

df.income_per_person = np.log(df.income_per_person)
#print df.head(5)
#print df.shape

#PERFORM PCA
# we need to scale data with mean = 0 and stdev = 1
pca = decomposition.PCA(n_components=2)
scaled_data = preprocessing.scale(df)
pca.fit(scaled_data)
transformed = pca.transform(scaled_data)

# let's take a look at the first ten rows of the transformed data

##print transformed[:10, :]

#Each row is a principle component(n). Each column is one of the original features.
#So this tells us the amount of weight, the components put on each of the original features.
#COMPONENTS

#print pca.components_

# HERE WE ARE TALKING ABOUT SOME OF THE PROPERTIES OF THE COMPONENTS.
#Components are orthogonal to each other (dot product = 0)

#print np.dot(pca.components_[0], pca.components_[1]) #Should be close to 0, means No correlation
#print (pca.components_.T**2).sum(axis=0) #Check to see if they have length of 1(apples to apples for Coeff). COMPONENT LOADING
#ALWAYS use the ROTATED component matrix!!

#print pca.explained_variance_ratio_ #For each features see how much of variation is explained by each part.


plt.figure(figsize=(12, 8))
sns.regplot(transformed[:, 0], transformed[:, 1], fit_reg=False)
feature_vectors = pca.components_.T
arrow_size, text_pos = 2.0, 2.5
for i, v in enumerate(feature_vectors):
    plt.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
              head_width=0.05, head_length=0.1)
    plt.text(v[0]*text_pos, v[1]*text_pos, df.columns[i], color='r', 
             ha='center', va='center', fontsize=16)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.title("PC plane with original feature projections.", fontsize=18)
plt.show()
'''
#DIFFERENCE IN COMPLETION RATES

# Let's take a look at the difference in completion rates: (male - female)
df_diff = df[["income_per_person", "employment", "life_expectancy"]].copy()
df_diff["diff_completion"] = df.completion_male - df.completion_female

pca_diff = decomposition.PCA(n_components=2)
scaled_diff = preprocessing.scale(df_diff)
pca_diff.fit(scaled_diff)
transformed_diff = pca_diff.transform(scaled_diff)
print transformed_diff[:10, :]

print pca_diff.components_

plt.figure(figsize=(12, 8))
sns.regplot(transformed_diff[:, 0], transformed_diff[:, 1], fit_reg=False)
feature_vectors_diff = pca_diff.components_.T
for i, v in enumerate(feature_vectors_diff):
    plt.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
              head_width=0.05, head_length=0.1)
    plt.text(v[0]*text_pos, v[1]*text_pos, df_diff.columns[i], color='r', 
             ha='center', va='center', fontsize=16)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.title("PC plane with original feature projections.", fontsize=18)
plt.show()

df_transform = pd.DataFrame(index=df.index, data=transformed_diff, columns=["PC1", "PC2"])

# merge dataframes based on index - "country"
# we now have a dataframe that contains both the components and the original features
df_transform = df_transform.join(df_diff)
'''
'''
# Example with modifications from:
# http://mpld3.github.io/examples/html_tooltips.html

# Define some CSS to control our custom labels

#css = '''
#table
#{
#  border-collapse: collapse;
#}
#th
#{
#  color: #ffffff;
#  background-color: #000000;
#}
#td
#{
#  background-color: #cccccc;
#}
#table, th, td
#{
#  font-family:Arial, Helvetica, sans-serif;
#  border: 1px solid black;
#  text-align: right;
#}
#'''

'''
fig, ax = plt.subplots(figsize=(10, 6))

labels = []
for i in range(len(df_transform)):
    label = df_transform.ix[[i], :].T
    label.columns = [df_transform.index[i]]
    # .to_html() is unicode; so make leading 'u' go away with str()
    labels.append(str(label.to_html()))

points = ax.plot(df_transform.PC1, df_transform.PC2, 'o', color='b',
                 mec='k', ms=15, mew=1, alpha=.6)


ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Interactive projection on PC plane', size=20)

tooltip = plugins.PointHTMLTooltip(points[0], labels,
                                   voffset=10, hoffset=10, css=css)
plugins.connect(fig, tooltip)

mpld3.display()
'''