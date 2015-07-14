import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib.pyplot as plt


df = pd.read_csv('turnstile_weather_v2.csv')
#I used the improved dataset for this project

def normalise(data):
    mean = data.mean()
    stdev = data.std()
    return (data - mean)/stdev

def linear_regression(features, values): 
    #features = sm.add_constant(features) Code needs to be tweaked to add constant
    results = sm.OLS(values, features).fit()
    params = results.params
    print results.summary()
    return params

def get_predictions(df):
	numerical_features = df[['rain','weekday']]
	features = normalise(numerical_features)
	station_dummies = pd.get_dummies(df['station'], prefix='Station')
	hour_dummies = pd.get_dummies(df['hour'],prefix='Hour')
	features = features.join(hour_dummies)
	features.drop(['Hour_0'],axis=1,inplace=True)
	features = features.join(station_dummies)
	features.drop(['Station_104 ST'],axis=1,inplace=True)
	values = df['ENTRIESn_hourly']
	params = linear_regression(features,values)
	predictions = np.dot(features, params)
	print numerical_features.corr()
	return predictions 	

def rain_mann_whitney_plus_means(df):
    Rain_mean=df['ENTRIESn_hourly'][df.rain==1].mean()
    NoRain_mean=df['ENTRIESn_hourly'][df.rain==0].mean()
    Rain_median=df['ENTRIESn_hourly'][df.rain==1].median()
    NoRain_median=df['ENTRIESn_hourly'][df.rain==0].median()
    Rain=df['ENTRIESn_hourly'][df.rain==1]
    NoRain=df['ENTRIESn_hourly'][df.rain==0]
    U,p=scipy.stats.mannwhitneyu(Rain,NoRain,use_continuity=True)
      
    print 'ENTRIESn_hourly with Rain Mean: %f' % Rain_mean
    print 'ENTRIESn_hourly with No Rain Mean: %f' % NoRain_mean
    print 'ENTRIESn_hourly with Rain Median: %f' % Rain_median
    print 'ENTRIESn_hourly with No Rain Median: %f' % NoRain_median
    print 'U Statistic:'
    print U
    print 'One-tailed p value: '
    print p
    print 'Two-tailed p value: '
    print p * 2
    return Rain_mean, NoRain_mean,U,p    

def rain_entries_histogram(df):
	plt.figure()
	df['ENTRIESn_hourly'][df.rain==0].hist(bins=75,label='NoRain',color='b')
	df['ENTRIESn_hourly'][df.rain==1].hist(bins=75,label='Rain',color='g')
	plt.title("NYC Subway Ridership Histogram")
	plt.ylabel("Frequency")
	plt.xlabel("ENTRIESn_Hourly")
	plt.legend()
	return plt.show()

#Count # of rain	
def stacked_rain_chart_count(df):
	df['count']=0
	w=df.groupby(['weekday','rain'],as_index=False)
	w=w.count().loc[:,['weekday','rain','count']]
	Rain=w['count'][w.rain==1]
	NoRain=w['count'][w.rain==0]
	group_labels = ['Weekend','Weekday']
	num_items = len(group_labels)
	ind = np.arange(num_items)
	width = .35
	p1=plt.bar(ind,NoRain,width,color='g',align='center')
	p2=plt.bar(ind,Rain,width,color='b', align='center', bottom = NoRain)
	plt.ylabel('Occurences')
	plt.title('Weather by DayType')
	plt.legend( (p1[0], p2[0]), ('NoRain', 'Rain'))
	plt.xticks(ind, group_labels)
	plt.xlim([min(ind) - 1,max(ind)+1 ])
	return plt.show()

#Count # of rain	
#Count the number of rows that has the same (weekday,rain) values
def stacked_rain_chart_sum(df):
	w1=df.groupby(['weekday','rain'],as_index=False)
	w=w1.sum().loc[:,['weekday','rain','ENTRIESn_hourly']]
	Rain=w['ENTRIESn_hourly'][w.rain==1]
	NoRain=w['ENTRIESn_hourly'][w.rain==0]
	group_labels = ['Weekend','Weekday']
	num_items = len(group_labels)
	ind = np.arange(num_items)
	width = .35
	p1=plt.bar(ind,NoRain,width,color='g',align='center')
	p2=plt.bar(ind,Rain,width,color='b', align='center', bottom = NoRain)
	plt.ylabel('ENTRIESn_hourly')
	plt.title('Ridership by DayType and Weather')
	plt.legend( (p1[0], p2[0]), ('NoRain', 'Rain'))
	plt.xticks(ind, group_labels)
	plt.xlim([min(ind) - 1,max(ind)+1 ])
	return plt.show()	


"""
I use groupby() to create a DataFrameGroupBy object. This object is similar to a python dict that maintains key-value pairs. Keys are the (weekday,rain) and the values are the row indexes of the original data frame that matches the key values.
Then I use count() to convert the DataFrameGroupBy object back to a DataFrame frame again. 
count() makes the values of each column becomes the number of entries that has 
the same (weekday,rain) value. If you want to sum up the values of each column 
that has the same (weekday,rain) values, you can use sum() to convert the 
DataFrameGroupBy object back to a DataFrame.

http://nbviewer.ipython.org/gist/kanhua/fe694f5d6391465af789

"""	
 
#Histogram and Stacked Bar as Subplots              
def turnsitle_visualizations(df):
	plt.figure()
	plt.subplot(211)
	df['ENTRIESn_hourly'][df.rain==0].hist(bins=75,label='NoRain',color='b')
	df['ENTRIESn_hourly'][df.rain==1].hist(bins=75,label='Rain',color='g')
	plt.title("NYC Subway Ridership Histogram")
	plt.ylabel("Frequency")
	plt.xlabel("ENTRIESn_Hourly")
	plt.legend()
	plt.subplot(212)
	df['count']=0
	w=df.groupby(['weekday','rain'],as_index=False) 
	w=w.count().loc[:,['weekday','rain','count']] #Counts occurences
	Rain=w['count'][w.rain==1]
	NoRain=w['count'][w.rain==0]
	group_labels = ['Weekend','Weekday']
	num_items = len(group_labels)
	ind = np.arange(num_items)
	width = .35
	p1=plt.bar(ind,NoRain,width,color='g',align='center')
	p2=plt.bar(ind,Rain,width,color='b', align='center', bottom = NoRain)
	plt.ylabel('Occurences')
	plt.title('Weather by DayType')
	plt.legend( (p1[0], p2[0]), ('NoRain', 'Rain'))
	plt.xticks(ind, group_labels)
	plt.xlim([min(ind) - 1,max(ind)+1 ])
	plt.subplots_adjust(left=None, bottom=None, right=None, top=.9, wspace=None, hspace=.5)
	return plt.show()

                 
get_predictions(df) 
#rain_mann_whitney_plus_means(df)
#rain_entries_histogram(df)
#stacked_rain_chart_count(df)
#stacked_rain_chart_sum(df)
#turnsitle_visualizations(df)	 
 