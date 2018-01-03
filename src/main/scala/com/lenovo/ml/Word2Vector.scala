package com.lenovo.ml

/**
  * Created by YangChenguang on 2017/10/17.
  */
import org.apache.spark.sql.SparkSession
import DataPreprocess.segWords
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline

object Word2Vector {
  def main(args:Array[String]): Unit = {
    // 1、创建Spark程序入口
    val sparkSession = SparkSession.builder().appName("Word2Vector").enableHiveSupport().getOrCreate()

    // 2、读取训练数据，对文本预处理后分词
    val tableName = args(0)
    val matrix = sparkSession.sql("SELECT text FROM " + tableName + " where text is not null")
    val words = segWords(sparkSession, args(1), args(2), args(3), args(4), matrix).repartition(6).cache()

    // 3、数据准备
    val tokenizer = new RegexTokenizer().setInputCol("words").setOutputCol("wordsArray")
    val remover = new StopWordsRemover().setInputCol("wordsArray").setOutputCol("filteredWords")

    // 4、训练Word2Vec模型
    val word2Vec = new Word2Vec().setInputCol("filteredWords").setOutputCol("features").setStepSize(0.025).setNumPartitions(1)
      .setMaxIter(1).setMaxSentenceLength(1000).setWindowSize(5).setVectorSize(args(5).toInt).setMinCount(10).setSeed(12345L)
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, word2Vec))
    val Word2VecModel = pipeline.fit(words)

    // 5、保存模型
    Word2VecModel.write.save(args(6))

    sparkSession.stop()
  }
}
