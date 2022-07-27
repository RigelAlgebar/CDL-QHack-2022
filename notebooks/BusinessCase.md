# Preamble
Clustering is the process of dividing the datasets into groups such that points in the same group are as similar as possible, and points in different groups are as dissimilar as possible. K-means clustering is an unsupervised Machine Learning (ML) protocol whose main goal is to classify datasets into K clusters. The term “k-means” was first used by James MacQueen in 1967, and published as a journal article in 1982. K-means clustering can minimize within cluster variances, the problem is computationally difficult (NP-hard) and the time complexity of the algorithm is specified by O(LNpK), where L = number of iterations taken to form clusters, N = number of datapoints, p = dataset dimension, K = centroids.
K-means clustering can be used in almost every domain, ranging from banking to recommendation engines, cyber security, document clustering to image segmentation. It is typically applied to data that has a smaller number of dimensions, is numeric, and is continuous and here are some examples:

*	Customer Segmentation: Clustering helps marketers improve their customer base, work on target areas. This segmentation helps companies target specific clusters/groups of customers for specific campaigns.
*	Document Classification: Cluster documents in multiple categories based on tags, topics, and the content of the document.
*	Delivery store optimization: Optimize the process of good delivery, use k-means to find the optimal number of launch locations.
*	Fraud detection: There different typs of fraud detection for example insurance fraud detection, transaction fraud detection. It is possible to isolate new claims based on its proximity to clusters that indicate fraudulent patterns.

Since either insurance fraud detection and transaction fraud detection can potentially have a multi-million dollar impact on a company, the ability to detect frauds is of crucial importance.

# Business: Improve Transaction Fraud Detection 
Today, Internet has become the essential component of life. Now a day’s shopping in home is liked by more and more people especially the young people. Apart from shopping buying virtual goods also become popular, for example, microtransaction in games. Credit cards are often used to do the purchase. As Credit card has the power to purchase the things, its frauds also increased. For example, Credit card fraud costs American companies billions of dollars each year. While consumers are never on the hook for fraudulent charges, reporting thefts and replacing cards can be aggravating and time consuming.

<img width="468" alt="bankcard" src="https://user-images.githubusercontent.com/79662449/181138035-a45c4579-27fd-4ca4-8f2d-283c8653cbef.png">

## Why it is important
Credit card fraud is a huge problem in the United States. In fact, it is estimated that \$16 billion was lost to credit card fraud in 2017. Executive Summary Overview In 2021, the Consumer Sentinel Network took in over 5.7 million reports, an increase from 2020. - Fraud: 2.8 million (49\%) of all reports) - Identity theft: 1.4 million (25\%) - Other: 1.5 million (27\%) In 2021, people filed more reports about Identity Theft (25.0\% of all reports), in all its various forms, than any other type of complaint. Imposter Scams, a subset of Fraud reports, followed with 984,756 reports from consumers in 2021 (17.2\% of all reports). Credit Bureaus, Information Furnishers and Report Users (10.3\% of all reports) rounded out the top three reports to Sentinel. Fraud There were over 984,000 imposter scam reports to Sentinel. Seventeen percent of those reported a dollar loss, totaling over \$2.3 billion lost to imposter scams in 2021. These scams include, for example, romance scams, people falsely claiming to be the government, a relative in distress, a well-known business, or a technical support expert, to get a consumer’s money. Of the nearly 2.8 million fraud reports, 25\% indicated money was lost. In 2021, people reported losing more than \$5.8 billion to fraud – an increase of \$2.4 billion over 2020.[1](https://www.cnet.com/personal-finance/credit-cards/credit-card-theft-is-the-problem-that-wont-go-away-it-just-changes/)

## A Solution: Using Hybrid K-Means Clustering Algorithm to Improve Fraud Detection
K-means clustering algorithm can divide the data set into k distinct non-overlapping partitioned clusters. In such scenario, K-means may form the simplest preliminary basis of fraud detection by segregating the data set into fraudulent and non-fraudulent sets. Some K-means experiments have been carried out on  accounting data set, and the results indicated high accuracy and significant relation between misrepresented values of the factors K-means clustering has been simulatedand fraud [2](https://link.springer.com/chapter/10.1007/978-981-33-4859-2_17), [3](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf). Hybrid K-Means consists quantum algoritm can significantly speed up the computational time.[4](https://arxiv.org/pdf/1909.04226.pdf)