/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.examples.knowledgebot

import java.io.File
import java.text.SimpleDateFormat
import java.util.Date

import com.huaban.analysis.jieba.JiebaSegmenter.SegMode
import com.huaban.analysis.jieba.{JiebaSegmenter, WordDictionary}
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.example.utils.SimpleTokenizer.{shaping, toTokens, vectorization}
import com.intel.analytics.bigdl.example.utils.WordMeta
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Loss, Top1Accuracy, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.zoo.examples.textclassification.TextClassification.{analyzeTexts, classNum, loadRawData, log}
import com.intel.analytics.zoo.examples.textclassification.TextClassificationParams
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import org.slf4j.{Logger, LoggerFactory}
import org.apache.log4j.{Level => Level4j, Logger => Logger4j}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{CountVectorizer, IDF, SQLTransformer, StopWordsRemover}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{BLAS, Vectors}
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode, SparkSession}
import scopt.OptionParser
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNEstimator}
import org.apache.spark.ml.Pipeline

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
import scala.reflect.io.Path

case class QuestionAnsweringParams(baseDir: String = "/home/arda/chentao/work/knowledgebotpoc/data",
                                   w2iFile: String = "/home/arda/chentao/work/knowledgebotpoc/data/cc.zh.300.vec",
                                   tokenLength: Int = 300,
                                   answerSeqLength: Int = 200,
                                   questionSeqLength: Int = 50,
                                   batchSize: Int = 40,
//                                   nbEpoch: Int = 80,
                                   nbEpoch: Int = 1,
                                   learningRate: Double = 0.001,
                                   partitionNum: Int = 4)


object QuestionAnswering {

  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level4j.INFO)

  private def getTokenizer(dictFile: String) = {
    val dictPath = java.nio.file.Paths.get(dictFile)
    WordDictionary.getInstance().loadUserDict(dictPath)

    val toTokens = (sentence: String) => {
      val segmenter = new JiebaSegmenter()
      segmenter.process(sentence, SegMode.INDEX).asScala.map(_.word).toArray
    }
    toTokens
  }

  def buildWord2Vec(word2Meta: Map[String, Int], tokenLength: Int, w2vFile: String): Map[Int, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = mutable.Map[Int, Array[Float]]()
    // val filename = s"/home/yang/sources/trainingMaterials/cc.zh.300.vec"
    val source = Source.fromFile(w2vFile)
    val iter = source.getLines()
    val Array(n, d) = iter.next().split(" ").map(_.toInt)
    for (i <- 0 until n) {
      val line = iter.next()
      val values = line.stripMargin.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1,  tokenLength + 1).map(_.toFloat)
        preWord2Vec.put(word2Meta(word), coefs)
      }
    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[QuestionAnsweringParams]("TextClassification Example") {
      opt[String]("baseDir")
        .text("The base directory containing the training data")
        .action((x, c) => c.copy(baseDir = x))
      opt[String]("w2iFile")
        .text("The file containing pretrained word vectors")
        .action((x, c) => c.copy(w2iFile = x))
      opt[Int]("partitionNum")
        .text("You may want to tune the partitionNum if you run in spark mode")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]("tokenLength")
        .text("The size of each word vector")
        .action((x, c) => c.copy(tokenLength = x))
      opt[Int]("answerSeqLength")
        .text("The length of a sequence")
        .action((x, c) => c.copy(answerSeqLength = x))
      opt[Int]("questionSeqLength")
        .text("The length of a sequence")
        .action((x, c) => c.copy(questionSeqLength = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]("nbEpoch")
        .text("The number of epoches to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[Double]("learningRate")
        .text("The learning rate for TextClassifier")
        .action((x, c) => c.copy(learningRate = x))

    }

    parser.parse(args, QuestionAnsweringParams()).map { param =>
      val conf = Engine.createSparkConf()
        .setAppName("Question answering")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      val spark = SQLContext.getOrCreate(sc)
      import spark.implicits._
      Engine.init

      val tokenLength = param.tokenLength
      val faqDir = s"${param.baseDir}/parquet/faq/"
      val articleDir = s"${param.baseDir}/parquet/dumparticle/"

      val dictFile = s"${param.baseDir}/pre-process-text-files/sougou.dict/"

      val toToken = getTokenizer(dictFile)

      spark.udf.register("to_token", toToken)

      val stopWordsFile = s"${param.baseDir}/pre-process-text-files/stopwords.txt/"
      val stopWords = Source.fromFile(stopWordsFile).getLines().toSet

      val isStop = (words: String) => {
        stopWords(words)
      }

      spark.udf.register("is_stop_word", isStop)


      val articleDf = spark.read.parquet(articleDir).repartition(param.partitionNum)

      val faqDf = spark.read.parquet(faqDir).repartition(param.partitionNum)

      faqDf.registerTempTable("faq_table")
      articleDf.registerTempTable("article_table")

      val faqWords = spark
        .sql("select explode(to_token(concat(question, answer))) as word from faq_table")
      val articleWords = spark
        .sql("select explode(to_token(concat(Title, Description, Content))) as word" +
          " from article_table")

      val w2i = faqWords.union(articleWords).distinct()
        .map(_.getString(0)).filter(!stopWords(_)).collect().zipWithIndex.toMap

      val w2v = buildWord2Vec(w2i, tokenLength, param.w2iFile)

      val toIndex = (words: mutable.WrappedArray[String]) => {
        words.map(w2i.get(_)).filter(_.isDefined).map(_.get + 1)
      }

      spark.udf.register("to_index", toIndex)

      val goldenData = faqDf.selectExpr("question as q", "answer as a")
        .union(articleDf.selectExpr("Title as q", "concat(Description, Content) as a"))
        .withColumn("label", lit(1.0))

      goldenData.registerTempTable("golden_table")

      val negativeDf = spark.sql("select t1.q as q, t2.a as a, 0.0 as label" +
        " from golden_table as t1, golden_table as t2 where t1.a <> t2.a")
        .sample(false, 0.001148071828083603, seed = 1l) //0.001148071828083603

      val pad = (arr: mutable.WrappedArray[Int], length: Int, padValue: Int) => {
        val result =
          if (length <= arr.length) {
            arr.slice(0, length)
          }
          else {
            (arr ++: Array.fill(length - arr.length)(padValue)).toSeq
          }
        result
      }

      spark.udf.register("pad", pad)

      val merge = (arr1: mutable.WrappedArray[Int], arr2: mutable.WrappedArray[Int]) => {
        arr1 ++ arr2
      }

      spark.udf.register("merge", merge)

      val toFloat = (arr: mutable.WrappedArray[Int]) => {
        arr.map(_.toFloat)
      }

      spark.udf.register("to_float", toFloat)

      val toEmbedding = (arr: mutable.WrappedArray[Int]) => {
        val result = arr.flatMap(w2v.getOrElse(_, Array.fill(param.tokenLength)(0.0f)))
        result
      }

      spark.udf.register("to_embedding", toEmbedding)

      val positiveCount = goldenData.count()
      val negativeCount = negativeDf.count()

      val data = goldenData.union(negativeDf)

      val Array(trainDf, valDf, testDf) = data.randomSplit(Array(0.6, 0.3, 0.1))

      val tokenizer = new SQLTransformer().setStatement("select q as raw_q, a as raw_a, to_token(q) q, to_token(a) as a, label from __THIS__")
      val wordsToIndices = new SQLTransformer().setStatement("select raw_q, raw_a, to_index(q) as q, to_index(a) as a, label from __THIS__")

      val padding = new SQLTransformer().setStatement(
        s"select raw_q, raw_a, pad(q, ${param.questionSeqLength}, ${w2i.size + 1}) as q," +
          s"pad(a, ${param.answerSeqLength}, ${w2i.size + 1}) as a, label from __THIS__")

      val merging = new SQLTransformer().setStatement("select raw_q, raw_a, to_embedding(merge(q, a)) as feature, label + 1.0 as label from __THIS__")


      val transformedValDf = Array(tokenizer, wordsToIndices, padding, merging)
        .foldLeft(valDf)((df, trans) => trans.transform(df))

      val model = buildModel2(w2i.size + 1, tokenLength)

      val dataLength = param.questionSeqLength + param.answerSeqLength
      val date = {
        val now = new Date()
        val dateFormat= new SimpleDateFormat("yyyy-MM-dd HH::mm::ss")
        dateFormat.format(now)
      }
      val estimator = NNClassifier[Float](model, ClassNLLCriterion[Float](), Array(dataLength, tokenLength))

      estimator.setFeaturesCol("feature")
      estimator.setLabelCol("label")
      estimator.setBatchSize(param.batchSize)
      estimator.setMaxEpoch(param.nbEpoch)
      estimator.setOptimMethod(new Adam[Float](param.learningRate))
      estimator.setValidation(Trigger.everyEpoch, transformedValDf, Array(new Top1Accuracy[Float](), new Loss[Float]()), param.batchSize)
      estimator.setTrainSummary(TrainSummary("./summaries", s"QuestionAnswering_$date"))
      estimator.setValidationSummary(ValidationSummary("./summaries", s"QuestionAnswering_$date"))

      val pipeline = new Pipeline().setStages(Array(tokenizer, wordsToIndices, padding, merging, estimator))

      val classifierModel = pipeline.fit(trainDf)

      val trainPred = classifierModel.transform(trainDf).cache()
      printStats(trainPred, "training")
      trainDf.write.mode(SaveMode.Overwrite).parquet(s"${param.baseDir}/result_train")

      val valPred = classifierModel.transform(valDf).cache()
      printStats(valPred, "validation")
      valPred.write.mode(SaveMode.Overwrite).parquet(s"${param.baseDir}/result_val")

      val testPred = classifierModel.transform(testDf).cache()
      printStats(testPred, "testing")
      testPred.write.mode(SaveMode.Overwrite).parquet(s"${param.baseDir}/result")
    }
  }

  def printStats(df: DataFrame, prefix: String) = {
    val recall = df.filter(col("label") === lit(2)).groupBy()
      .agg(sum((col("label") === col("prediction")).cast("Double")).as("correct"), count(lit(1.0)).as("all"))
      .select(col("correct") / col("all"))
      .first().get(0)

    val precision = df.filter(col("prediction") === lit(2)).groupBy()
      .agg(sum((col("label") === col("prediction")).cast("Double")).as("correct"), count(lit(1.0)).as("all"))
      .select(col("correct") / col("all"))
      .first().get(0)

    val accuracy = df.groupBy()
      .agg(sum((col("label") === col("prediction")).cast("Double")).as("correct"), count(lit(1.0)).as("all"))
      .select(col("correct") / col("all"))
      .first().get(0)

    println(s"[$prefix] Accuracy: $accuracy, precision: $precision, recall: $recall")
  }



  def buildModel(inputEmbeddingDim: Int, embeddingDim: Int) = {
    val deep = Sequential()
    val lookupTable = LookupTable(inputEmbeddingDim, embeddingDim)
    lookupTable.setWeightsBias(Array(Tensor[Float](inputEmbeddingDim, embeddingDim).randn(0, 0.1)))
    //    deep.add(lookupTable)
    deep.add(Reshape(Array(250, embeddingDim))) // 250
    deep.add(TemporalConvolution(embeddingDim, 128, 5))
    deep.add(ReLU())
    deep.add(TemporalMaxPooling(5, 5)) // 25
    deep.add(Reshape(Array(128 * 49)))
    deep.add(Linear(128 * 49, 100)).add(ReLU())
      .add(Linear(100, 75)).add(ReLU())
      .add(Linear(75, 50)).add(ReLU())
      .add(Linear(50, 25)).add(ReLU())
      .add(Linear(25, 2))
      .add(LogSoftMax())
    deep
  }

  def buildModel3(inputEmbeddingDim: Int, embeddingDim: Int) = {
    val deep = Sequential()
    val lookupTable = LookupTable(inputEmbeddingDim, embeddingDim)
    lookupTable.setWeightsBias(Array(Tensor[Float](inputEmbeddingDim, embeddingDim).randn(0, 0.1)))
    //    deep.add(lookupTable)
    //    deep.add(Transpose(Array((2, 3))))
    //    deep.add(Contiguous())
    deep.add(Reshape(Array(embeddingDim, 1, 250))) // 250
    deep.add(SpatialConvolution(embeddingDim, 128, 25, 1, padH = -1, padW = -1))
    deep.add(ReLU())
    deep.add(SpatialMaxPooling(25, 1, 25, 1, padH = -1, padW = -1)) // 25
    deep.add(SpatialConvolution(128, 128, 10, 1, padH = -1, padW = -1)) // 25
    deep.add(ReLU())
    deep.add(SpatialMaxPooling(10, 1, 10, 1, padH = -1, padW = -1)) // 5
    deep.add(Reshape(Array(128)))
    deep.add(Linear(128, 100)).add(ReLU())
      .add(Linear(100, 75)).add(ReLU())
      .add(Linear(75, 50)).add(ReLU())
      .add(Linear(50, 25)).add(ReLU())
      .add(Linear(25, 2))
      .add(LogSoftMax())
    deep
  }

  def buildModel2(inputEmbeddingDim: Int, embeddingDim: Int) = {
    val deep = Sequential()
    val lookupTable = LookupTable(inputEmbeddingDim, embeddingDim)
    lookupTable.setWeightsBias(Array(Tensor[Float](inputEmbeddingDim, embeddingDim).randn(0, 0.1)))
    // deep.add(lookupTable)
    // deep.add(deepColumn)
    val concat = Concat(2)
    val q = Sequential()
    q.add(Narrow(-2, 1, 50))
    q.add(Reshape(Array(50, embeddingDim)))
    q.add(TemporalConvolution(embeddingDim, 32, 5))
    q.add(ReLU())
    q.add(TemporalMaxPooling(50 - 5 + 1)) // 25
    q.add(Reshape(Array(32)))

    val a = Sequential()
    a.add(Narrow(-2, 51, 200))
    a.add(Reshape(Array(200, embeddingDim)))
    a.add(TemporalConvolution(embeddingDim, 32, 5))
    a.add(ReLU())
    a.add(TemporalMaxPooling(200 - 5 + 1)) // 25
    a.add(Reshape(Array(32)))

    concat.add(q)
    concat.add(a)

    deep.add(concat)
    //    deep.add(Linear(128, 64)).add(ReLU())
    //      .add(Linear(64, 32)).add(ReLU())
    //      .add(Linear(32, 2))
    //      .add(LogSoftMax())
    //    deep.add(Reshape(Array(250 * 300)))
    deep.add(Linear(64, 50)).add(ReLU())
      //      .add(Linear(100, 75)).add(ReLU())
      //      .add(Linear(75, 50)).add(ReLU())
      .add(Linear(50, 25)).add(ReLU())
      .add(Linear(25, 2))
      .add(LogSoftMax())
    deep
  }

}

object DumpDataToParquet {
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Question answering")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    val spark = SQLContext.getOrCreate(sc)
    import scala.collection.JavaConverters._
    import spark.implicits._


    val faqDataPath = Path(s"${args(0)}/faq")
    val faqFiles = faqDataPath.walk.filter(_.isFile).map(_.path).toSeq

    val faqDataDf = spark.read.json(faqFiles :_*).flatMap { r =>
      val answer = r.getAs[String](0)
      val questions = r.getList[String](1).asScala
      questions.map((_, answer))
    }.toDF("question", "answer")

    faqDataDf.repartition(1).write.mode(SaveMode.Overwrite).parquet(s"${args(1)}/parquet/faq")

    val articleDataPath = Path(s"${args(0)}/dumparticle")
    val files = articleDataPath.walk.filter(_.isFile).map(_.path).toSeq

    val dataDf = spark.read.json(files :_*)

    dataDf.repartition(1).write.mode(SaveMode.Overwrite).parquet(s"${args(1)}/parquet/dumparticle")
  }
}
