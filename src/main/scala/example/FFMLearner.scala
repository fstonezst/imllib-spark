package example

import com.intel.imllib.ffm.classification.{FFMModel, FFMWithAdag}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by peter on 2017/8/11.
  */
object FFMLearner extends appEnv with Logging {

  var modelConfigFilePath = ""

  def main(args: Array[String]): Unit = {
    modelConfigFilePath = args(0)
    val sc = initSparkContext("FFM_Experiment")
    var sqlCon = initSparkContextHiveSql(sc)
    val featureData = getData(sqlCon, sc)

    val trainData = dataTransform(featureData.na.drop(), sqlCon, sc)

    ffmLearn(trainData, true, true, 2, 10, 0.2, 0)

    sc.stop()
  }


  /**
    * ffm模型训练函数
    * 输入指定格式数据进行ffm训练,normalization默认为True，random默认为false
    * @param data 训练数据  格式：[(filedIndex,featureIndex,Value),...]
    * @param globalBias bias
    * @param onewayInteractions 是否加入原始特征权重
    * @param k 隐因子
    * @param iters epoch
    * @param eta 权值更新步长
    * @param lam 正则参数
    */
  def ffmLearn(data: RDD[(Double, Array[(Int, Int, Double)])], globalBias: Boolean, onewayInteractions: Boolean, k: Int, iters: Int, eta: Double, lam: Int): Unit = {

    data.repartition(16).persist(StorageLevel.MEMORY_ONLY)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (training: RDD[(Double, Array[(Int, Int, Double)])], testing: RDD[(Double, Array[(Int, Int, Double)])]) = (splits(0), splits(1))

    val m = data.flatMap(x => x._2).map(_._1).collect.reduceLeft(_ max _) + 1

    val n = data.flatMap(x => x._2).map(_._2).collect.reduceLeft(_ max _)

    val ffm: FFMModel = FFMWithAdag.train(training, m, n, dim = (globalBias, onewayInteractions, k), n_iters = iters,
      eta = eta, lambda = lam, normalization = true, random = false, "adagrad")

    val scores: RDD[(Double, Double)] = testing.map(x => {
      val p = ffm.predict(x._2)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })

    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()
    logInfo(s"accuracy = $accuracy")
    //    ffm.save(sc, args(7))
    //    val sameffm = FFMModel.load(sc, args(7))

  }


  /**
    * 读取特征对应的field id
    * 从配置文件中读取每个特征对应的fieldIndex，未找到的特征对应的fieldIndex默认为0
    * @param columns 训练数据特征
    * @param confPath 配置文件路径
    * @return [特征名->fieldIndex]
    */
  def readField(columns: Array[String], confPath: String): Map[String, String] = {
    val configInfo = FileUtil.parseJsonConfig(confPath)
    val field = configInfo.apply("field")
    var fieldMap: Map[String, String] = StringUtil.parseJson(field)
    val colSet: Set[String] = columns.toSet[String]
    for (key <- colSet)
      if (!fieldMap.keySet.contains(key))
        logWarning(s"===feature ${key} is not in fieldMap will be group to 0 field===")
    fieldMap
  }


  /**
    * 将DataFrame格式数据转换为imllib-ffm可用格式数据
    * @param df 训练数据 值不可为Null
    * @param sqlCon
    * @param sc
    * @return [label,[(fieldIndex,featureIndex,value)..]]
    */
  def dataTransform(df: DataFrame, sqlCon: HiveContext, sc: SparkContext): RDD[(Double, Array[(Int, Int, Double)])] = {
    var encodeData: DataFrame = df

    var fieldMap: Map[String, String] = readField(df.columns, modelConfigFilePath)


    var vctSet: Set[String] = Set()

    for (key <- fieldMap.keySet) {
      if (encodeData.columns.indexOf(key) != -1) {

        val indexer = new StringIndexer()
          .setInputCol(key)
          .setOutputCol(key + "Index")
          .fit(encodeData)

        val indexed = indexer.transform(encodeData)

        val encoder = new OneHotEncoder()
          .setInputCol(key + "Index")
          .setOutputCol(key + "Vct")
          .setDropLast(false)


        encodeData = encoder.transform(indexed)

        vctSet += (key + "Vct")
        fieldMap = fieldMap + ((key + "Index") -> fieldMap(key))
        fieldMap = fieldMap + ((key + "Vct") -> fieldMap(key))

        encodeData = encodeData.drop(key + "Index").drop(key)

      }
    }

    val boradColName = sc.broadcast(encodeData.columns)
    var colNameArr: Array[String] = boradColName.value

    val res = encodeData.map(
      line => {
        val label: Int = line.getAs[Int]("click")
        val y: Double = if (label > 0) 1.0 else 0.0

        val nodeArray: ArrayBuffer[(Int, Int, Double)] = ArrayBuffer()
        var featureIndex: Int = 1
        for (i <- 1 until colNameArr.length) {
          var colName: String = colNameArr(i)

          if (vctSet.contains(colName)) {
            val vct: SparseVector = line.getAs[SparseVector](colName)

            if (fieldMap.contains(colName)) {
              nodeArray += ((fieldMap.apply(colName).toInt, (featureIndex + vct.indices(0)), 1.0))
            } else {
              nodeArray += ((0, (featureIndex + vct.indices(0)), 1.0))
            }
            featureIndex += (vct.size + 1)
          } else {
            if (fieldMap.contains(colName)) {
              nodeArray += ((fieldMap.apply(colName).toInt, featureIndex, line.getAs[Double](colName)))
            } else {
              nodeArray += ((0, featureIndex, line.getAs[Double](colName)))
            }
            featureIndex += 1
          }
        }
        val array: Array[(Int, Int, Double)] = nodeArray.toArray[(Int, Int, Double)]
        (y, array)
      }
    )
    res
  }

  /**
    * 从Hive中获取训练数据
    * @param sqlCon
    * @param sc
    * @return DataFrame
    */
  def getData(sqlCon: HiveContext, sc: SparkContext): DataFrame = {
    var date = FunctionUtil.getBeforeToday(-1)
    /*var featureData = sqlCon.sql(
      s"""
         |select
         |  click,fcid,cid,cid0,cid1,cid2,shopid,shoplevel,position
         |  ,platform
         |  ,price,price_min,price_max,item_ctr1,uv,item_favs,favs_14d
         |  ,item_pay_number_7,item_pay_number_30,item_pay_number_leiji
         |  ,item_pay_gmv_7,item_pay_gmv_30,item_pay_gmv_leiji,return_rate
         |  ,evaluate_orders
         |  ,discount,discount_price,online_num,fans_new,fans,order_cnt,back_uv
         |  ,shop_ctr1,avg_pay,avg_price,convert_rate,xinpin_num
         |  ,return_orders,shop_uv,shop_pv,shop_desc_dsr_avg,shop_quality_score
         |  ,shop_price_score,shop_service_score,complain_rate,impression_all_1d
         |  ,click_all_1d,impression_all_4d,click_all_4d,impression_all_7d,click_all_7d
         |  ,shop_impression_all_1d,shop_click_all_1d,shop_impression_all_4d,shop_click_all_4d
         |  ,shop_impression_all_7d,shop_click_all_7d,impression_fcid_1d,click_fcid_1d
         |  ,impression_fcid_4d,click_fcid_4d,impression_fcid_7d,click_fcid_7d,hour,userfcid
         |  ,user_itemcnt,user_clickcnt,carttag30d,carttag14d
         |  ,shop_fcidimpression_4d,shop_fcidclick_4d,shop_fcidctr_4d,shop_fcidimpression_1d
         |  ,shop_fcidclick_1d,shop_fcidctr_1d,bid
         |
         |
         |  --,cartnum14d,cartnum30d,fcid_ctr,fcid_pos_ctr,fcid_pos_click,fcid_click,fcid_pos_impression,fcid_impression
         |  --,item_price_score,item_quality_score
         |from
         |  ad_algo_cpc_app_acm_feature_v2_wall
         |where
         |  visit_date = '$date'
         |  and channel = '1'
         |  and tradeitemid is not null
         |  and title is not null
         |  and fcid is not null
         |  and position < 300
       """.stripMargin
    )*/

    var featureData = sqlCon.sql(
      s"""
         |select
         |  click,fcid,cid,cid0,cid1,cid2,shopid,shoplevel,position,platform
         |from
         |  ad_algo_cpc_app_acm_feature_v2_wall
         |where
         |  visit_date = '$date'
         |  and channel = '1'
         |  and tradeitemid is not null
         |  and title is not null
         |  and fcid is not null
         |  and position < 300
       """.stripMargin
    )
    featureData
  }

}
