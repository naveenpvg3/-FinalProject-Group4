print("CLUSTERING:\n")

print("We identified the metrics to use for clustering as follows: For each customer ID, we calculated TotalExpense, "
      "number of purchases, average amount per purchase, number of unique items bought. With the help of these metrics"
      "we will be able to categorize the customers into groups: High spending customers, regular (loyal) customers, etc. \n")

cluster_df1 = df1[['Total Expenditure']].merge(transactions, how='left', on='CustomerID')
cluster_df1['Avg. value per purchase'] = cluster_df1['Total Expenditure']/cluster_df1['Total Purchases']
cluster_df1 = cluster_df1.merge(unique_items, how='left', on='CustomerID')

print(cluster_df1.head())

X = cluster_df1[['Total Expenditure', 'Total Purchases', 'Avg. value per purchase','No. of unique items']]
ar= ['Total Expenditure', 'Total Purchases', 'Avg. value per purchase','No. of unique items']

print("Plotting SSE against various values of k")
sum_sqEr = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, init='random', n_init=10, max_iter=10)
    kmeans.fit(X)
    sum_sqEr.append(kmeans.inertia_)

f, axes = plt.subplots(1,1,figsize=(8,8))
plt.plot(range(1, 11), sum_sqEr)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors")
plt.show()

print("Using Elbow method, we can decide k = 4 as an appropriate number of clusters.")

kmeans = KMeans(n_clusters=4, random_state=42, init='random', n_init=10, max_iter=10)
kmeans.fit(X)
cluster_df1['Cluster']=kmeans.predict(X)

for i in range(3):
    for j in range(i+1,4):
        sns.scatterplot(x=ar[i], y=ar[j], hue='Cluster', data=cluster_df1, palette='tab10')
        plt.show()

print("From the scatter plots showing different clusters, we can conclude a few things and assign some characteristics to the clusters, \n")
print("Cluster 0: Total expenditure is the least of all, total purchases are minimal and a minimal number of unique items are bought by them.")
print("Cluster 1: Least total expenditure is observed. Total purchases are moderate. These customers have bought a large number of unique items.")
print("Cluster 2: Total expenditure is more than clusters 0 and 1, number of unique items bought is on a higher side as compared to the first two clusters.")
print("Cluster 3: Total expenditure is the maximum though the total purchases don’t vary much. \n")

for i in range(4):
      cluster_df1.boxplot(column=ar[i], by=["Cluster"])
      plt.show()

print("We plotted boxplots to confirm our observations from the scatter plots. \n")
print("Based on these observations we can categorize these clusters as follows: \n")
print("Cluster 0 and 1 need to be focussed more on, in terms of discounts and other offers in order to increase their purchase numbers..")
print("Cluster 2 consists of regular value but loyal customers who visit the store pretty often.")
print("Cluster 3 can be considered as high-valued customers for whom a loyalty program should be rolled out. These customers are loyal as well seeing the number of purchases.")
