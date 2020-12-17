#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas
import math
uni = pandas.read_csv(r'C:\Users\svega866\Desktop\cwurData.csv')
print(uni.columns.values)

selected_university = uni[uni["institution"] == "New York University"].iloc[0]
                                 
distance_columns = ['national_rank' ,'quality_of_education', 'alumni_employment', 'quality_of_faculty',
 'publications', 'influence', 'citations', 'broad_impact', 'patents' ,'score']
                            
def euclidean_distance(row):
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_university[k]) ** 2
    return math.sqrt(inner_value)    

nyu_distance = uni.apply(euclidean_distance, axis = 1)

uni_numeric = uni [distance_columns]

uni_normalized = (uni_numeric - uni_numeric.mean()) / uni_numeric.std()


from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor

uni_normalized.fillna(0, inplace = True)
nyu_normalized = uni_normalized [uni["institution"] == "New York University"]

euclidean_distances = uni_normalized.apply(lambda row: distance.euclidean(row, nyu_normalized), axis=1)

distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort_values("dist", inplace=True)

second_smallest = distance_frame.iloc[1] ["idx"]
most_similar_to_nyu = uni.loc[int(second_smallest)] ["institution"]

import random
from numpy.random import permutation

random_indices = permutation(uni.index)
test_cutoff = math.floor(len(uni)/4)

test = uni.loc[random_indices[1:test_cutoff]]
train = uni.loc[random_indices[test_cutoff:]]

x_columns = ['national_rank' ,'quality_of_education', 'alumni_employment', 'quality_of_faculty',
 'publications', 'influence', 'citations', 'broad_impact', 'patents']

y_column = ['score']


knn = KNeighborsRegressor(n_neighbors=5) 
knn.fit(train[x_columns], train[y_column])
predictions = knn.predict(test[x_columns])

actual = test[y_column]

mse = (((predictions - actual) ** 2).sum()) / len(predictions)

mse


# In[ ]:





# In[ ]:




