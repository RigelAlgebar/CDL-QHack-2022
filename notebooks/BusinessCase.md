# Preamble
Clustering is the process of dividing the datasets into groups such that points in the same group are as similar as possible and points in different groups are as dissimilar as possible. K-means clustering is an unsupervised Machine Learning (ML) protocol whose main goal is to classify datasets into K clusters. The term “k-means” was first used by James MacQueen in 1967, and published as a journal article in 1982. K-means clustering algorithms minimize within cluster variances. The problem is computationally difficult (NP-hard), and the time complexity of the algorithm is specified by O(LNpK), where L = number of iterations taken to form clusters, N = number of datapoints, p = dataset dimension, K = centroids.
K-means clustering can be used in almost every domain, ranging from banking to recommendation engines, cyber security, document clustering to image segmentation. It is typically applied to data that has a smaller number of dimensions, is numeric, and is continuous and here are some examples:

*	Customer Segmentation: Clustering helps marketers improve their customer base and work on target areas. This segmentation helps companies target specific clusters/groups of customers for specific campaigns.
*	Document Classification: Cluster documents in multiple categories based on tags, topics, and the content of the document.
*	Delivery store optimization: Optimize the process of good delivery by using k-means to find the optimal number of launch locations.
*	Fraud detection: There are different types of fraud detection; for example insurance fraud detection, transaction fraud detection. It is possible to isolate new claims based on its proximity to clusters that indicate fraudulent patterns.

Since either insurance fraud detection and transaction fraud detection can potentially have a multi-million dollar impact on a company, the ability to detect fraud is of crucial importance.

# Business: Improve Transaction Fraud Detection 
Today, Internet has become the essential component of life. Now a days shopping in home is liked by more and more people, especially the young people. Apart from shopping, buying virtual goods has also become popular, for example, microtransaction in games. Credit cards are often used to do the purchase. As Credit card has the power to purchase the things, its fraudulent use has also increased. For example, Credit card fraud costs American companies billions of dollars each year. While consumers are never on the hook for fraudulent charges, reporting thefts and replacing cards can be aggravating and time consuming for them.

<img width="468" alt="bankcard" src="https://user-images.githubusercontent.com/79662449/181138035-a45c4579-27fd-4ca4-8f2d-283c8653cbef.png">

## Why it is important
Credit card fraud is a huge problem in the United States. In fact, it is estimated that \$16 billion was lost to credit card fraud in 2017. In 2021, the Consumer Sentinel Network took in over 5.7 million reports, an increase from 2020. Fraud accounted for 2.8 million or 49\% of all reports and Identity theft, 1.4 million or 25\%. Of the nearly 2.8 million fraud reports, 25\% indicated money was lost. In 2021, people reported losing more than \$5.8 billion to fraud – an increase of \$2.4 billion over 2020.[1](https://www.cnet.com/personal-finance/credit-cards/credit-card-theft-is-the-problem-that-wont-go-away-it-just-changes/). The Nilson report estimatee the total losses to bank card fraud over the next ten years to be $400 billion.[2](https://nilsonreport.com/upload/content_promo/NilsonReport_Issue1209.pdf)

 

## A Solution: Using Hybrid K-Means Clustering Algorithm to Improve Fraud Detection
K-means clustering algorithm can divide the data set into k distinct non-overlapping partitioned clusters. In such scenario, K-means may form the simplest preliminary basis of fraud detection by segregating the data set into fraudulent and non-fraudulent sets. Some K-means experiments have been carried out on  accounting data set, and the results indicated high accuracy and significant relation between misrepresented values of the factors K-means clustering has been simulatedand fraud [3](https://link.springer.com/chapter/10.1007/978-981-33-4859-2_17), [4](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf). Hybrid K-Means consists of a quantum algoritm that can significantly speed up the computations.[5](https://arxiv.org/pdf/1909.04226.pdf)
