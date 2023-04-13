#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


#loading dataset
data = pd.read_csv("C:\\Users\\Swetanshu's\\Downloads\\archive\\Instagram data.csv", encoding = 'latin1')
print(data.head())


# In[3]:


#checking any value is null
data.isnull().sum()


# In[4]:


#drpoing if any value is null
data = data.dropna()


# In[5]:


#checking data information
data.info()


# In[6]:


#Analyzing Instagram Reach
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()


# In[10]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()


# In[12]:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()


# In[13]:


#Analyzing Content

text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[14]:


text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[15]:


#Analyzing Relationships

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


# In[13]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()


# In[16]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()


# In[15]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()


# In[17]:


correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))


# In[18]:


conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# In[19]:


figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()


# In[20]:


x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[21]:


model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[22]:


# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[281.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)


# In[23]:


# Convert text data into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
text_features = vectorizer.fit_transform(data['Caption'] + ' ' + data['Hashtags']).toarray()


# In[24]:


# Concatenate numerical features and text features
numerical_features = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
features = np.concatenate([numerical_features, text_features], axis=1)


# In[25]:


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, data['Impressions'].values, test_size=0.2, random_state=42)


# In[26]:


# Fit a regression model
model = PassiveAggressiveRegressor()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"R^2 score: {score}")


# In[30]:


# Make a prediction
text = "This is a sample caption #sample #test"
text_features = vectorizer.transform([text]).toarray()
numerical_features = np.array([[100, 50, 5, 2, 20, 10]])
features = np.concatenate([numerical_features, text_features], axis=1)
prediction = model.predict(features)
print(f"Prediction: {prediction}")


# In[ ]:




