package org.apache.spark.mllib.classification

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer
import scala.io.Source


/**
 * A standard neural network implementation trained with back propagation.
 * Fully compatible with the source code of Spark / mlib 1.2.0
 * Using MNIST as a benckmark : average precision >= 0.99 for 2000 iterations.
 * Training time depends on the setting of Spark cluster.
 */
class NeuralNetwork( val inputUnit : Int,
                     val outputUnit : Int,
                     val hiddenLayer : Int,
                     val hiddenUnit : Int,
                     val model : List[Matrix]
                     ) extends ClassificationModel with Serializable {
  import NeuralNetworkWithSGD._
  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.map(data => predict(data))
  }

  override def predict(testData: Vector): Double = {
    maxIdx(forwardPropagate(model, testData).last)
  }

  private def maxIdx(output : Vector) = {
    val array = output.toArray
    array.indexOf(array.max)
  }
}

case class LabeledVector(val features : Vector, val label : Vector)

object NeuralNetworkWithSGD {
  def logisticFunction(theta : Vector, x : Vector) : Double = {
    1.0 / (1 + Math.exp(-(theta.toBreeze dot x.toBreeze)))
  }

  def logisticFunction(v : Double) = {
    1.0 / (1 + Math.exp(-v))
  }

  /**
   * @param model list of weight matrix on each layer, including the intercept (mapping to the bias term 1.0)
   * @param input the vector which contain a bias term 1.0
   * @return list of activations on each layer, including the input layer
   */
  def forwardPropagate(model : List[Matrix], input : Vector) : List[Vector] = {
    val initial = input.toBreeze.toDenseVector
    val activation = ListBuffer(initial)
    model.foldLeft(initial)((x, mat) => {
      val z : BDM[Double] = (mat.toBreeze.toDenseMatrix * x.toDenseMatrix.t)
      val a : BDV[Double] = z.toDenseVector.map{ logisticFunction(_) }
      activation += BDV.vertcat(BDV(1.0), a)  // plus bias unit
      activation.last
    })

    // remove bias unit for output layer
    val s = activation.size - 1
    val last = activation(s)
    activation.update(s, last(1 until last.length))

    activation.map(Vectors.fromBreeze(_)).toList
  }

  def cost(model : List[Matrix], trainingData : List[LabeledVector], lamda : Double) : Double = {
    - (1.0 / trainingData.size) * trainingData.map(cost(model, _)).sum  +
      (lamda / (2.0 * trainingData.size)) * model.map(_.toArray.map(e => e * e).sum).sum // regularizationTerm
  }

  private def cost(model : List[Matrix], trainingData : LabeledVector) : Double = {
    val expected = trainingData.label
    val actual = forwardPropagate(model, trainingData.features).last
    elementWiseOp(expected, actual, (e, a) => e * Math.log(a) + (1 - e) * Math.log(1 - a)).toArray.sum
  }

  private def elementWiseOp(a : Vector, b : Vector, op : (Double, Double) => Double) : Vector = {
    Vectors.dense(a.toArray.zip(b.toArray).map(z => op(z._1, z._2)))
  }

  def gradient2(model : List[BDM[Double]], trainingData : LabeledVector, lamda : Double) : List[BDM[Double]] = {
    val activation = forwardPropagate(
      model map {m => Matrices.fromBreeze(m)},
      trainingData.features).map(_.toBreeze.toDenseVector)
    var error = ListBuffer(activation.last - trainingData.label.toBreeze)
    var lastErr = error(0)
    for(i <- 1 until model.size reverse) {
      val derivative = activation(i) :* (BVOnes(activation(i).size) - activation(i))
      val t = model(i).t * lastErr.toDenseMatrix.t

      lastErr = t.toDenseVector :* derivative
      lastErr = lastErr(1 until lastErr.size)
      error += lastErr
    }
    // add error term for the first layer ( but it will never be used )
    error += BDV()
    error = error.reverse

    val gradients = (0 until model.size) map { i =>
      error(i+1).toDenseMatrix.t * activation(i).toDenseMatrix
    } toList

    // add regularization term
    val regularization = (0 until model.size) map {
      i => {
        val nRow = model(i).rows
        val nCol = model(i).cols
        val m = BDM.ones[Double](nRow, nCol)
        m(::, 0) := BDV.zeros[Double](nRow)
        m :*= lamda
        m :*= model(i)
        m
      }
    }
    (0 until model.size) foreach { i => gradients(i) :+= regularization(i) }
    gradients
  }


  // use one training example to compute the partial derivative term
  def gradient(model : List[Matrix], trainingData : LabeledVector, lamda : Double) : List[Matrix] = {
    val activation = forwardPropagate(model, trainingData.features).map(_.toBreeze.toDenseVector)
    var error = ListBuffer(activation.last - trainingData.label.toBreeze)
    var lastErr = error(0)
    for(i <- 1 until model.size reverse) {
      val derivative = activation(i) :* (BVOnes(activation(i).size) - activation(i))
      val t = model(i).toBreeze.toDenseMatrix.t * lastErr.toDenseMatrix.t

      lastErr = t.toDenseVector :* derivative
      lastErr = lastErr(1 until lastErr.size)
      error += lastErr
    }
    // add error term for the first layer ( but it will never be used )
    error += BDV()
    error = error.reverse

    val gradients = (0 until model.size) map { i =>
      error(i+1).toDenseMatrix.t * activation(i).toDenseMatrix
    } toList

    // add regularization term
    val regularization = (0 until model.size) map {
      i => {
        val nRow = model(i).numRows
        val nCol = model(i).numCols
        val m = BDM.ones[Double](nRow, nCol)
        m(::, 0) := BDV.zeros[Double](nRow)
        m :*= lamda
        m :*= (model(i).toBreeze.toDenseMatrix)
        m
      }
    }
    (0 until model.size) foreach { i => gradients(i) :+= regularization(i) }

    gradients map { Matrices.fromBreeze(_) }
  }

  private def BVOnes(size : Int) : BV[Double] = {
    BDV.apply(1 to size map (_.toDouble) toArray)
  }

  def train( input:RDD[LabeledVector],
             numIterations: Int,
             learningRate: Double,
             inputUnit : Int,
             outputUnit : Int,
             hiddenUnit : Int,
             hiddenLayer : Int,
             lamda : Double,
             initialWeights: List[Matrix]): NeuralNetwork = {
    parallelizedSGD(input, numIterations, learningRate, inputUnit, outputUnit, hiddenUnit, hiddenLayer, lamda, 0.005, initialWeights)

  }


  // Parallelized Stochastic Gradient Descent.
  // TODO replaced by Spark-specific gradient and update sub functions
  private def parallelizedSGD(
                               input:RDD[LabeledVector],
                               numIterations: Int,
                               learningRate: Double,
                               inputUnit : Int,
                               outputUnit : Int,
                               hiddenUnit : Int,
                               hiddenLayer : Int,
                               lamda : Double,
                               miniBatchFraction : Double,
                               initialWeights: List[Matrix]): NeuralNetwork = {

    var weights = initialWeights map { _.toBreeze.toDenseMatrix }
    val emptyGradients = initialWeights map {
      weightMatrix => BDM.zeros[Double](weightMatrix.numRows, weightMatrix.numCols)
    }

    for (i <- 1 to numIterations) {
      val bcWeights = input.context.broadcast(weights)
      val (gradientSum, miniBatchSize) = input.sample(false, miniBatchFraction, 42 + i).aggregate((emptyGradients, 0L)) (
        (c, v) => {
          val g = gradient2(bcWeights.value, v, lamda)
          for(j <- 0 until c._1.size) {
            (c._1)(j) :+= g(j)
          }

          (c._1, c._2 + 1)
        },

        (c1, c2) => {
          for(j <- 0 until c1._1.size) {
            (c1._1)(j) :+= (c2._1)(j)
          }
          (c1._1, c1._2 + c2._2)
        }
      )

      // batch update
      for(j <- 0 until gradientSum.size) {
        gradientSum(j) :/= miniBatchSize.toDouble
        gradientSum(j) :*= learningRate
        weights(j) :-= gradientSum(j)
      }
    }

    new NeuralNetwork(inputUnit, outputUnit, hiddenLayer, hiddenUnit, weights map { Matrices.fromBreeze(_) })
  }

  // train sequentially with statistic gradient descent in memory in a single node
  private def trainSequentially(input:RDD[LabeledVector],
                                numIterations: Int,
                                learningRate: Double,
                                inputUnit : Int,
                                outputUnit : Int,
                                hiddenUnit : Int,
                                hiddenLayer : Int,
                                lamda : Double,
                                initialWeights: List[Matrix]): NeuralNetwork = {
    var model = initialWeights
    for(i <- 1 to numIterations) {
      println("total traing samples = " + input.count())

      input.collect().foreach(
        trainingData => {
          model = update(model, gradient(model, trainingData, lamda), learningRate)
        }
      )
    }

    new NeuralNetwork(inputUnit, outputUnit, hiddenLayer, hiddenUnit, model)
  }

  private def update(model : List[Matrix], delta : List[Matrix], learningRate : Double) = {
    (0 until model.size) map {
      i => {
        Matrices.fromBreeze(model(i).toBreeze - (delta(i).toBreeze :*= learningRate))
      }
    } toList
  }
}
