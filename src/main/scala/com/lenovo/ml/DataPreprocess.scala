package com.lenovo.ml

import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import scala.collection.mutable
import scala.util.matching.Regex
import org.ansj.library.DicLibrary
import org.ansj.recognition.impl.StopRecognition
import org.ansj.splitWord.analysis.DicAnalysis

/**
  * Created by YangChenguang on 2017/12/27.
  */
object DataPreprocess {
  def textCleaner(rawText: DataFrame): Dataset[String] = {
    // 过滤文本中的时间、网址和邮箱
    val regex1 = new Regex("""[-—0-9a-z]+[:]+[0-9a-z]+[:]?""")
    val regex2 = new Regex("""[0-9]+年|[0-9]+月|[0-9]+[日]|[0-9]+[天]|[0-9]+[号]|[0-9]+[次]""")
    val regex3 = new Regex("""http[s]?://[a-z0-9./?=_-]+""")
    val regex4 = new Regex("""[0-9_a-z]+([-+.][0-9_a-z]+)*@[0-9_a-z]+([-.][0-9_a-z]+)*\.[0-9_a-z]+([-.][0-9_a-z]+)*""")

    rawText.map(x => x.toString).map(x => x.substring(1,x.length - 1).toLowerCase).map(x => regex1.replaceAllIn(x,""))
      .map(x => regex2.replaceAllIn(x,"")).map(x => regex3.replaceAllIn(x,"")).map(x => regex4.replaceAllIn(x,""))
  }

  def segWords(sparkSession: SparkSession, stopWordsPath: String, dictionaryPath: String, synonymWordsPath: String,
               singleWordsPath: String, rawText: DataFrame): DataFrame = {
    val filter = new StopRecognition()
    // 设定停用词性
    filter.insertStopNatures("w","ns","nr","t","r","u","e","y","o")
    // 加载停用词表
    val stopWords = sparkSession.sparkContext.textFile(stopWordsPath).cache()
    stopWords.collect().foreach{line => filter.insertStopWords(line)}
    // 加载自定义词表
    val dictionary = sparkSession.sparkContext.textFile(dictionaryPath).cache()
    dictionary.collect().foreach{line => DicLibrary.insert(DicLibrary.DEFAULT, line)}
    stopWords.collect().foreach{line => DicLibrary.insert(DicLibrary.DEFAULT, line)}
    // 构建同义词表
    val synonymWords = sparkSession.sparkContext.textFile(synonymWordsPath).cache()
    var synonymMap: Map[String, String] = Map()
    synonymWords.collect().foreach{line =>
      val data = line.split(" ",2)
      synonymMap = synonymMap + (data(0) -> data(1))
    }
    // 构建单字白名单
    val singleWords = sparkSession.sparkContext.textFile(singleWordsPath).cache()
    val singleWhiteList: mutable.Set[String] = mutable.Set()
    singleWords.collect().foreach{line => singleWhiteList.add(line)}

    // 通过广播将词表发送给各节点
    val stop = sparkSession.sparkContext.broadcast(filter)
    val dic = sparkSession.sparkContext.broadcast(DicLibrary.get(DicLibrary.DEFAULT))
    val synonym = sparkSession.sparkContext.broadcast(synonymMap)
    val single = sparkSession.sparkContext.broadcast(singleWhiteList)

    // 读取文本数据，过滤后分词
    textCleaner(rawText).map { x =>
      val parse = DicAnalysis.parse(x, dic.value).recognition(stop.value)
      // 抽取分词结果，不附带词性
      val words = for(i<-Range(0,parse.size())) yield parse.get(i).getName
      val filterWords = words.map(_.trim).filter(x => x.length > 1 || single.value.contains(x))
      filterWords.map(x => if(synonym.value.contains(x)) synonym.value(x) else x).mkString(" ")
    }.toDF("words")
  }
}
