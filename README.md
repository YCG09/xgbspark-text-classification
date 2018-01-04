XGBoost on Spark for Chinese Text Classification

* Data Source: Hive
* Word Segmentation: ANSJ
* Feature Engineering: NGram + TF-IDF or Word2Vec
* Classification Algorithm: XGBoost
* Model Selection and Tuning: Cross Validation + Grid Search

## Environments

* [Spark](http://spark.apache.org)  2.1.1
* [Hive](https://hive.apache.org)  1.2.1
* [XGBoost4J-Spark](https://github.com/dmlc/xgboost/tree/master/jvm-packages)  0.7
* [Ansj](https://github.com/NLPchina/ansj_seg)  5.1.2
