package com.lenovo.ml

/**
  * Created by YangChenguang on 2017/9/15.
  */
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.StructType
import DataPreprocess.segWords
import org.apache.spark.ml.PipelineModel

object XGBoostInference {
  def main(args:Array[String]): Unit = {
    // 1、创建Spark程序入口
    val sparkSession = SparkSession.builder().appName("XGBoostInference").enableHiveSupport().getOrCreate()

    // 2、读取训练数据，对文本预处理后分词
    val tableName = args(0)
    val matrix = sparkSession.sql("SELECT * FROM " + tableName)
    val words = segWords(sparkSession, args(1), args(2), args(3), args(4), matrix.select("text"))

    // 3、将原数据与分词结果关联起来
    val rows = matrix.rdd.zip(words.rdd).map{
      case (rowLeft, rowRight) => Row.fromSeq(rowLeft.toSeq ++ rowRight.toSeq)
    }
    val schema = StructType(matrix.schema.fields ++ words.schema.fields)
    val matrixMerge = sparkSession.createDataFrame(rows, schema)

    // 4、构建特征向量
    val featuredModelTrained = sparkSession.sparkContext.broadcast(PipelineModel.read.load(args(5)))
    val dataPrepared = featuredModelTrained.value.transform(matrixMerge).repartition(18).cache()

    // 5、加载分类模型，产出故障预测结果
    val xgbModelTrained = sparkSession.sparkContext.broadcast(PipelineModel.read.load(args(6)))
    val prediction = xgbModelTrained.value.transform(dataPrepared)

    // 6、将预测结果写到HDFS
    prediction.select("caseid", "text", "filteredWords", "predictedLabel", "probabilities").rdd.coalesce(1).saveAsTextFile(args(7))

    sparkSession.stop()
  }
}
