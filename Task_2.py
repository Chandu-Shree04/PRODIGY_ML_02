import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA

#Loading the data file
data_file= pd.read_csv('C:/Users/Admin/Downloads/Mall_Customers.csv')

#removing irrelevent columns from the dataset
data_file.drop(columns=['CustomerID'],inplace=True)

data_file['Gender']. replace({'Male':1, 'Female':0}, inplace=True)

#filling missing values
data_file.fillna(0,inplace=True)

#perorming k means clustering
n_clusters=3
kmeans=KMeans(n_clusters=n_clusters)
kmeans.fit(data_file)

#adding cluster labels to the dataframe
data_file['Cluster']=kmeans.labels_

#printing cluster centroid
print("Cluster Centroid:")
print(pd.DataFrame(kmeans.cluster_centers_,columns=data_file.columns[:-1]))

#printing cluster sizes
print("Cluster sizes:")
print(data_file['Cluster'].value_counts())

#Visualizing Clusters
pca=PCA(n_components=2)
df_pca=pca.fit_transform(data_file.drop(columns=['Cluster']))

plt.figure(figsize=(10,8))
for cluster in range(n_clusters):
    cluster_label=""
    if cluster==0:
        cluster_label="Minimum Spenders"
    elif cluster==1:
        cluster_label="Medium Spenders"
    elif cluster==2:
        cluster_label="Maximum Spenders"

    plt.scatter(df_pca[data_file['Cluster']==cluster,0],df_pca[data_file['Cluster']==cluster, 1],label=cluster_label)

#Plotting cluster centroids
centroid_pca=pca.transform(kmeans.cluster_centers_)
plt.scatter(centroid_pca[:,0],centroid_pca[:,1],marker='x',s=200,c='black',label='centroids')

plt.title('K-Means clustering of customers based on Purchase history')
plt.xlabel('PC1-Purchase Behaviour')
plt.ylabel('PC2-Spending Habits')
plt.legend()
plt.grid(True)
plt.show()


