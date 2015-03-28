package org.apache.spark.mllib.classification

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vectors, Matrices, Matrix}
import org.apache.spark.rdd.RDD

import scala.io.Source

/**
 * A example to classify MNIST dataset with spark standalone mode.
 * download mnist.csv : https://www.kaggle.com/c/digit-recognizer/download/train.csv
 */
class Example {
  val dataPath = "mnist.csv"
  val conf = new SparkConf().setAppName("ann").setMaster("local")
  val sc = new SparkContext(conf)

  def main(args : Array[String]) = {
    multiClassify
  }

  def multiClassify = {
    // Load data
    val data = Source.fromFile(dataPath).getLines().toList.filterNot(_.startsWith("label"))

    // Take the first 80% as training samples, the remaining 20% as test samples
    // To make the example simpler, it only uses samples with label = 0 or label = 1,
    // making it a binary classification
    // Also, to make it simple, no cross-validation or feature scaling is applied
    val trainSet = data.take((data.size * .8).toInt).map(extract(_)).filter(p => p._2 == 0 || p._2 == 1)
    val testSet = data.drop((data.size * .8).toInt).map(extract(_)).filter(p => p._2 == 0 || p._2 == 1)
    println("training set size = " + trainSet.size)
    println("testing set size = " + testSet.size)

    // Initialize neural network and start training
    val initialWeights = List(randomParams(3, 785), randomParams(2, 4))
    val classifier = NeuralNetworkWithSGD.train(
      input = toLabeledVector(trainSet),
      numIterations = 1000,  // number of total iterations. here it uses batch SGD
      learningRate = 0.1, // parameter of learning rate
      lamda = 0.00, // parameter of the regularization term
      initialWeights = initialWeights,
      hiddenLayer = 1, hiddenUnit = 3, inputUnit = 785, outputUnit = 10 // parameters of standard ANN
    )
    println("model = " + classifier.model)


    // Start testing
    val test = toLabeledVector(testSet)
    val predicted = test.map(t => classifier.predict(t.features))
    println(predicted)

    val actual = predicted
    val expected = test.map(_.label).map(_.toArray.indexOf(1).toDouble)

    println("actual = ")
    println(actual.collect().foreach(n => print(n + ", ")))
    println("expected = ")
    println(expected.collect().foreach(n => print(n + ", ")))

    println("total size of test samples = " + test.count())
    println("error number = " + error(actual.collect(), expected.collect()))
  }

  // helper functions
  def error(actual : Array[Double], expected : Array[Double]) = {
    0 until actual.size map { i => if(actual(i) == expected(i)) 0 else 1 } sum
  }

  def equal(actual : List[Int], expected : List[Int]) = {
    actual.indexOf(1) == expected.indexOf(1)
  }

  def toResult(p : Array[Double]) : List[Int] = {
    val maxIdx = p.indexOf(p.max)
    (0 to 1) map (i => if(i == maxIdx) 1 else 0) toList
  }

  def randomParams(row : Int, col : Int) : Matrix = {
    Matrices.dense(row, col, (1 to (row * col) map (_ => Math.random() - 0.5)) toArray)
  }

  def digitVec(n : Int) : Array[Double] = {
    (0 to 1).toArray.map(t => if(t == n) 1.0 else 0.0)
  }

  def extract(xy : String) : (Array[Double], Int) = {
    val nums = xy.split(",").toArray.map(_.toDouble)
    (nums.drop(1).map(t => (t - 125) / 255).+:(1.0), nums(0).toInt)   // a simple feature scaling
  }

  def toLabeledVector(data : List[(Array[Double], Int)]) : RDD[LabeledVector] = {
    val labeledVectors = data.map(d => LabeledVector(Vectors.dense(d._1), Vectors.dense(digitVec(d._2))))
    sc.parallelize(labeledVectors)
  }
}