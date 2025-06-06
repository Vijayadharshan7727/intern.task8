# intern.task8
This project demonstrates how to perform customer segmentation using the K-Means clustering algorithm on a mall customer dataset. The goal is to group customers based on their demographics and spending behavior, which can help businesses target marketing strategies more effectively.

Dataset

The dataset used is Mall_Customers.csv, which contains the following columns:

CustomerID — Unique identifier for each customer (dropped in preprocessing)
Gender — Gender of the customer (Male/Female)
Age — Age of the customer
Annual Income (k$) — Annual income of the customer in thousand dollars
Spending Score (1-100) — Score assigned by the mall based on customer spending behavior
Features

Gender is encoded to numerical values: Male = 0, Female = 1
Data is scaled using StandardScaler for normalization
PCA (Principal Component Analysis) is applied to reduce dimensionality to 2 components for visualization
The elbow method is used to determine the optimal number of clusters k
K-Means clustering is applied with the optimal k
Silhouette score is computed to evaluate cluster quality
Results are visualized using scatter plots of PCA components colored by cluster assignment
How to run

Clone the repository
Place the Mall_Customers.csv file in the project directory
Run the Python script (e.g. python kmeans_mall_customers.py)
Dependencies

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
You can install the dependencies using pip:

pip install pandas numpy matplotlib seaborn scikit-learn
Code explanation

Data preprocessing: The script reads the data, removes the CustomerID column, and encodes the gender as numeric.
Scaling: Features are standardized for better clustering performance.
PCA: Dimensionality is reduced to 2 principal components to help visualize clusters.
Elbow method: Calculates inertia for different values of k to identify the optimal number of clusters.
K-Means clustering: Performs clustering with the chosen k and assigns cluster labels.
Visualization: Shows clusters on a 2D scatter plot of PCA components.
Evaluation: Prints the silhouette score to assess how well clusters are separated.
Output

Elbow plot to choose the optimal number of clusters
PCA scatter plot colored by cluster
Silhouette score printed in the console
