package com.lenovo.ml

/**
  * Created by YangChenguang on 2017/9/14.
  */
import org.apache.spark.SparkException
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import DataPreprocess.segWords
import scala.collection.mutable
import org.apache.spark.ml.feature._
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostEstimator}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}

object XGBoostTrain {
  def featureEngineeringTFIDF(sparkSession: SparkSession, dataMatrix: DataFrame, savePath: String): DataFrame ={
    // 获取nGram
    val tokenizer = new RegexTokenizer().setInputCol("words").setOutputCol("wordsArray")
    val remover = new StopWordsRemover().setInputCol("wordsArray").setOutputCol("filteredWords")
    val nGram2 = new NGram().setN(2).setInputCol("filteredWords").setOutputCol("gram-2")
    val nGram3 = new NGram().setN(3).setInputCol("filteredWords").setOutputCol("gram-3")

    // 计算TF-IDF
    val countVectorizer_1gram = new CountVectorizer().setInputCol("filteredWords")
    val countVectorizer_2gram = new CountVectorizer().setInputCol("gram-2")
    val countVectorizer_3gram = new CountVectorizer().setInputCol("gram-3")
    val idf_1gram = new IDF().setInputCol(countVectorizer_1gram.getOutputCol).setOutputCol("tfidf-1gram").setMinDocFreq(10)
    val idf_2gram = new IDF().setInputCol(countVectorizer_2gram.getOutputCol).setOutputCol("tfidf-2gram").setMinDocFreq(10)
    val idf_3gram = new IDF().setInputCol(countVectorizer_3gram.getOutputCol).setOutputCol("tfidf-3gram").setMinDocFreq(10)
    val assembler = new VectorAssembler().setInputCols(Array("tfidf-1gram", "tfidf-2gram", "tfidf-3gram")).setOutputCol("features")

    // 构造特征向量
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, nGram2, nGram3, countVectorizer_1gram,
      countVectorizer_2gram, countVectorizer_3gram, idf_1gram, idf_2gram, idf_3gram, assembler))
    pipeline.fit(dataMatrix).write.save(savePath)
    val pipelineModelTrained = sparkSession.sparkContext.broadcast(PipelineModel.read.load(savePath))
    pipelineModelTrained.value.transform(dataMatrix)
  }

  def featureEngineeringWord2Vec(sparkSession: SparkSession, dataMatrix: DataFrame, savePath: String): DataFrame ={
    // 加载预训练的Word2Vec模型，构造特征向量
    val pipelineModelTrained = sparkSession.sparkContext.broadcast(PipelineModel.read.load(savePath))
    pipelineModelTrained.value.transform(dataMatrix)
  }

  def crossValidation(xgboostParam: Map[String, Any], labelIndexer: StringIndexerModel,
                      evaluator: MulticlassClassificationEvaluator, trainingData: DataFrame): TrainValidationSplitModel = {
    // XGBoost Pipeline Model
    val xgbEstimator = new XGBoostEstimator(xgboostParam).setLabelCol("labelIndex").setFeaturesCol("features").setPredictionCol("prediction")
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val pipeline = new Pipeline().setStages(Array(xgbEstimator, labelConverter))

    // Grid Search + Cross Validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.eta, Array(0.08, 0.1))
      .addGrid(xgbEstimator.round, Array(50, 100))
      .addGrid(xgbEstimator.maxDepth, Array(300, 500))
      .build()
    val crossValidator = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)
    crossValidator.fit(trainingData)
  }

  def main(args:Array[String]): Unit ={
    // 1、创建Spark程序入口
    val sparkSession = SparkSession.builder().appName("XGBoostTrain").enableHiveSupport().getOrCreate()

    // 2、读取训练数据，对文本预处理后分词
    val tableName = args(0)
    val matrix = sparkSession.sql("SELECT * FROM " + tableName + " where text is not null")
    val words = segWords(sparkSession, args(1), args(2), args(3), args(4), matrix.select("text"))

    // 3、将原数据与分词结果关联起来
    val rows = matrix.rdd.zip(words.rdd).map{
      case (rowLeft, rowRight) => Row.fromSeq(rowLeft.toSeq ++ rowRight.toSeq)
    }
    val schema = StructType(matrix.schema.fields ++ words.schema.fields)
    val matrixMerge = sparkSession.createDataFrame(rows, schema)

    // 4、构建特征向量
    var featuredData = sparkSession.emptyDataFrame
    if (args(5).toLowerCase == "tfidf")
      featuredData = featureEngineeringTFIDF(sparkSession, matrixMerge, args(6))
    else if (args(5).toLowerCase == "word2vec")
      featuredData = featureEngineeringWord2Vec(sparkSession, matrixMerge, args(6))
    else
      throw new SparkException("Feature engineering algorithm must be TFIDF or Word2Vec")

    // 5、将label转化为数值
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex").fit(featuredData)
    val dataPrepared = labelIndexer.transform(featuredData).select("text", "features", "label", "labelIndex")

    // 6、按比例划分训练数据和测试数据
    val testSize = args(7).toDouble
    val splits = dataPrepared.randomSplit(Array(1 - testSize, testSize), seed = 12345L)
    val (trainingData, testData) = (splits(0).repartition(18).cache(), splits(1).repartition(18).cache())

    // 7、定义模型评估方法
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("labelIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")  // Spark2.0以前为"precision"

    // 8、设定模型参数，训练XGBoost文本分类模型
    val paramMap = new mutable.HashMap[String, Any]()
    paramMap += "nworkers" -> 18
    paramMap += "use_external_memory" -> false
    // paramMap += "eta" -> 0.1f
    // paramMap += "num_round" -> 50
    // paramMap += "max_depth" -> 300
    paramMap += "min_child_weight" -> 3
    paramMap += "alpha" -> 0.01
    paramMap += "gamma" -> 0
    paramMap += "subsample" -> 0.8
    paramMap += "colsample_bytree" -> 0.8
    paramMap += "scale_pos_weight" -> 1
    paramMap += "num_class" -> args(8).toInt
    paramMap += "objective" -> "multi:softprob"
    paramMap += "numEarlyStoppingRounds" -> 0
    paramMap += "trainTestRatio" -> 0.9
    paramMap += "booster" -> "dart"
    paramMap += "rate_drop" -> 0.1
    paramMap += "skip_drop" -> 0.5
    paramMap += "seed" -> 12345L
    val cvModel = crossValidation(paramMap.toMap, labelIndexer, evaluator, trainingData)

    // 9、分类模型的保存与加载
    val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    bestPipelineModel.write.save(args(9))
    val xgbModelTrained = sparkSession.sparkContext.broadcast(PipelineModel.read.load(args(9)))

    // 10、使用训练好的模型对测试集样本进行分类
    val prediction = xgbModelTrained.value.transform(testData)

    // 11、评估模型效果
    prediction.select("text", "label", "predictedLabel", "probabilities").rdd.coalesce(1).saveAsTextFile(args(10))
    val accuracy = evaluator.evaluate(prediction)
    sparkSession.sparkContext.parallelize(List("Accuracy = " + accuracy)).coalesce(1).saveAsTextFile(args(11))

    // 12、保存模型参数
    val stages = xgbModelTrained.value.stages
    val modelTrainingStage = stages(0).asInstanceOf[XGBoostClassificationModel]
    sparkSession.sparkContext.makeRDD(modelTrainingStage.extractParamMap().toSeq).coalesce(1).saveAsTextFile(args(12))

    sparkSession.stop()
  }
}
