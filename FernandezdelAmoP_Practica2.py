# PABLO FERNÁNDEZ DEL AMO      GEOMETRÍA COMPUTACIONAL PRÁCTICA 2

import numpy as np
from sklearn.cluster import KMeans, DBSCAN #importing KMeans from scikit-learn library
from sklearn import metrics
from scipy.spatial import ConvexHull, convex_hull_plot_2d #importing ConvexHull and convex_hull_plot_2d from scipy.spatial library
from scipy.spatial import Voronoi, voronoi_plot_2d #importing Voronoi and voronoi_plot_2d from scipy.spatial library
import matplotlib.pyplot as plt #importing pyplot from matplotlib library

# #############################################################################
# Here we define the system X of 1500 elements (people) with two states

archivo1 = "D:\Documents\MATEMATICAS_UCM\MATEMATICAS_22_23\CUATRI_2\GEOMETRIA_COMPUTACIONAL\Practicas\Practica_2\Personas_en_la_facultad_matematicas.txt" #path to file containing data
archivo2 = "D:\Documents\MATEMATICAS_UCM\MATEMATICAS_22_23\CUATRI_2\GEOMETRIA_COMPUTACIONAL\Practicas\Practica_2\Grados_en_la_facultad_matematicas.txt" #path to file containing labels
X = np.loadtxt(archivo1) #loading data into numpy array
Y = np.loadtxt(archivo2) #loading labels into numpy array

#If we wanted to standardize the values of the system, we would:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  


# #############################################################################
# Clustering using the KMeans algorithm
n_clusters=np.arange(2,16) #defining the range of clusters to test
list_silhouette=[]
for k in n_clusters:
   kmeans=KMeans(n_clusters=k, n_init = 'auto', random_state=0).fit(X) #clustering data using KMeans algorithm with k clusters
   labels = kmeans.labels_ #assigning labels to each data point
   silhouette = metrics.silhouette_score(X, labels) #computing silhouette score for the clustering
   print(f'For {k} clusters using KMEANS the Silhouette Coefficient is: {silhouette:.3f}') #printing silhouette score
   list_silhouette.append(silhouette) #adding silhouette score to a list

optimal_cluster = n_clusters[np.argmax(list_silhouette)]
plt.plot(n_clusters, list_silhouette)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for KMeans Clustering')
plt.axvline(x=optimal_cluster, linestyle='--', color='red', label='Optimal Cluster')
plt.legend()
plt.show()

#Random initialization of centroids


# #############################################################################
# Predicting elements to belong to a cluster:
kmeans = KMeans(n_clusters=optimal_cluster, n_init ='auto', random_state=0).fit(X) #performing KMeans clustering with 3 clusters
problem = np.array([[0, 0], [0, -1]]) #defining two new data points to be classified
clases_pred = kmeans.predict(problem) #predicting the cluster to which each new data point belongs
print(f'Los clusters usando KMEANS a los que pertenecen la persona con valores [0,0] y [0,-1] son respectivamente: {clases_pred}') #
vor=Voronoi(kmeans.cluster_centers_)

# #############################################################################
# Representamos el resultado con un plot
labels = kmeans.labels_
unique_labels = set(labels)

fig, ax = plt.subplots(figsize=(10, 5))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(range(len(unique_labels)), colors):
    if k == -1:
        col = [0, 0, 0, 1] # black for noise points
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5, label = f'Cluster {k}')
    
    
# Plot Voronoi diagram
voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_width=1, line_colors='gray')

ax.plot(problem[0,0],problem[0,1],'*', markersize=10, markerfacecolor="red", label = '[0,0]')
ax.plot(problem[1,0],problem[1,1], 'h', markersize=10, markerfacecolor="red", label = '[0,-1]')
plt.xlim(-2.5,2.5)
plt.ylim(-2.2,1.75)
ax.set_title("Optimal Numberof Clusters using KMeans = 3", fontsize = 10)
plt.suptitle("KMeans Clustering and Voronoi Tessellation", fontsize = 16 , y = 1)
plt.legend(loc='best')
plt.show()



# Apartado 2:

# define the range of epsilon values to try
epsilon = np.arange(0.1, 0.4, 0.01)

# initialize empty lists to store the results
sil_coeff_eucl = []
sil_coeff_manh = []
nclusters_eucl = []
nclusters_manh = []

# loop through each epsilon value
for i in epsilon:
    
    # fit DBSCAN models using euclidean and manhattan distance metrics
    dbeuc = DBSCAN(eps=i, min_samples=10, metric='euclidean').fit(X)
    dbmanh = DBSCAN(eps=i, min_samples=10, metric='manhattan').fit(X)
    
    # find the core samples for each model
    core_samples_mask = np.zeros_like(dbeuc.labels_, dtype=bool)
    core_samples_mask[dbeuc.core_sample_indices_] = True
    core_samples_mask = np.zeros_like(dbmanh.labels_, dtype=bool)
    core_samples_mask[dbmanh.core_sample_indices_] = True
    
    # extract the cluster labels from each model
    labelseucl = dbeuc.labels_
    labelsmanh = dbmanh.labels_
    
    # determine the number of clusters for each model, ignoring noise if present
    nclusters_eucl.append(len(set(labelseucl)) - (1 if -1 in labelseucl else 0))
    nclusters_manh.append(len(set(labelsmanh)) - (1 if -1 in labelsmanh else 0))
    
    # count the number of noise points for each model
    n_noise_ = list(labelseucl).count(-1)
    n_noise_ = list(labelsmanh).count(-1)
    
    # calculate the silhouette score for each model
    silhouette_eucl = metrics.silhouette_score(X, labelseucl)
    sil_coeff_eucl.append(silhouette_eucl)
    silhouette_manh = metrics.silhouette_score(X, labelsmanh)
    sil_coeff_manh.append(silhouette_manh)
    print(f'DBSCAN: The silhouette coefficient using the euclidean and manhattan metrics  for the umbral threshold {i:.3f} are:{silhouette_eucl:.3f}, {silhouette_manh:.3f}')


# Plot Silhouette Coefficient
fig, ax = plt.subplots()
ax.plot(epsilon, sil_coeff_eucl, marker='o', markersize=5, label='Euclidean')
ax.plot(epsilon, sil_coeff_manh, marker='o', markersize=5, label='Manhattan')
ax.set_xlabel('Distance Threshold')
ax.set_ylabel('Silhouette Coefficient')
ax.set_title('Silhouette Coefficient vs Distance Threshold')
ax.legend(title='Metric Used')
plt.show()

# Plot number of clusters
fig, ax = plt.subplots()
ax.plot(epsilon, nclusters_eucl, marker='o', markersize=5, label='Euclidean')
ax.plot(epsilon, nclusters_manh, marker='o', markersize=5, label='Manhattan')
ax.set_xlabel('Distance Threshold')
ax.set_ylabel('Number of Clusters')
ax.set_title('Number of Clusters vs Distance Threshold')
ax.legend(title='Metric Used')
plt.show()

# Plot Clusters vs Silhouette each method:
fig, ax = plt.subplots()
ax.plot(nclusters_eucl, sil_coeff_eucl, 'o', markersize=4,  label = 'DBSCAN Euclidean')
ax.plot(nclusters_manh, sil_coeff_manh, 'o', markersize=4, label = 'DBSCAN Manhattan')
ax.plot(n_clusters, list_silhouette, 'o',  markersize=4, label = 'KMEANS')
ax.legend(title='CLustering Method')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Coefficient')
ax.set_title('Number of Clusters vs Silhouette Coefficient')
plt.xticks(range(0, 19, 1))
plt.show()

# Find the maximum silhouette coefficient value
optimo = max(sil_coeff_eucl)

# Apply DBSCAN clustering algorithm with epsilon parameter determined from the optimal silhouette coefficient value
# Set minimum number of samples to be considered a core point to 10
# Use Euclidean distance as the distance metric
db = DBSCAN(eps=epsilon[sil_coeff_eucl.index(optimo)], min_samples=10, metric='euclidean').fit(X)

# Obtain the labels assigned to each data point by DBSCAN
labels = db.labels_

# Get the unique cluster labels
unique_labels = set(labels)

# Assign a color to each cluster label
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# Create a figure to visualize the clusters
plt.figure(figsize=(8, 4))

# For each cluster label, plot the corresponding data points with the assigned color
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Use black color for noise points
        col = [0, 0, 0, 1]

    # Find the data points that belong to the current cluster label
    class_member_mask = (labels == k)

    # Plot the core data points with a larger marker size
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    # Plot the non-core data points with a smaller marker size
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

# Set the title of the plot
plt.title('Estimated number of optimal DBSCAN clusters: 1')

# Display the plot
plt.show()
