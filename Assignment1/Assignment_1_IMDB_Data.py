
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('/home/abhishek/Downloads/movie_metadata.csv')
data_raw = data

# Correlation matrix ------------------------------------------------------------------------------------

corr = data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
# 7 scatter plots on the dataset pairs that have the highest correlation --------------------------------------

data1 = data.loc[data.actor_1_facebook_likes < 125000]
plt.figure(2,figsize=(10,10))
plt.scatter(data1.actor_1_facebook_likes,data1.cast_total_facebook_likes)
plt.xlabel('actor_1_facebook_likes')
plt.ylabel('cast_total_facebook_likes')


plt.figure(3,figsize=(10,10))
plt.scatter(data.num_voted_users,data.num_user_for_reviews)
plt.xlabel('num_voted_users')
plt.ylabel('num_user_for_reviews')

plt.figure(4,figsize=(10,10))
plt.scatter(data.num_critic_for_reviews,data.movie_facebook_likes)
plt.xlabel('num_critic_for_reviews')
plt.ylabel('movie_facebook_likes')

data2 = data.loc[data.actor_2_facebook_likes < 40000]
plt.figure(5,figsize=(10,10))
plt.scatter(data2.actor_2_facebook_likes,data2.cast_total_facebook_likes)
plt.xlabel('actor_2_facebook_likes')
plt.ylabel('cast_total_facebook_likes')

plt.figure(6,figsize=(10,10))
plt.scatter(data.num_voted_users,data.gross)
plt.xlabel('num_voted_users')
plt.ylabel('gross')

plt.figure(7,figsize=(10,10))
plt.scatter(data.num_voted_users,data.num_critic_for_reviews)
plt.xlabel('num_voted_users')
plt.ylabel('num_critic_for_reviews')

plt.figure(8,figsize=(10,10))
plt.scatter(data.num_user_for_reviews,data.num_critic_for_reviews)
plt.xlabel('num_user_for_reviews')
plt.ylabel('num_critic_for_reviews')
plt.show()

# We can also derive other insights like the following:

# 1. Directors that have the highest imdb rated films--------------------------------------------------------------

data3 = data.groupby('director_name').mean().reset_index()

data3 = data3.sort_values('imdb_score',ascending=0).reset_index()
data3["num_of_films"] = np.zeros((data3.shape[0],1))
for i in range(200):
    data3.num_of_films[i] = data.director_name[data["director_name"] == data3.director_name[i]].count() 
data3 = data3.loc[data3.num_of_films > 2]
print("Directors who have more than 2 films" , data3.shape[1])

# We find that when the directors are arrangered with average imdb rating of their film , only 19 directors out of
# 200 have more than 2 films.
data3 = data3.sort_values('num_of_films',ascending=0)
data3_show = data3[['director_name','imdb_score','num_of_films']]
#This shows the directors who have been consistent in producing films that have a high imdb score
print(data3_show.head(10))

# 2. Actors that are more likely to get high budeget films-----------------------------------------------------------

data = data_raw
data4 = data.groupby('actor_1_name').mean().reset_index()

data4 = data4.sort_values('budget',ascending=0).reset_index()

data4["num_of_films"] = np.zeros((data4.shape[0],1))
for i in range(200):
    data4.num_of_films[i] = data.actor_1_name[data["actor_1_name"] == data4.actor_1_name[i]].count() 

data4 = data4.loc[data4.num_of_films > 5]

data4_show = data4[['actor_1_name','budget','num_of_films']]
#This gives the actors who have the highest average of the film budget
print(data4_show.head(10))









