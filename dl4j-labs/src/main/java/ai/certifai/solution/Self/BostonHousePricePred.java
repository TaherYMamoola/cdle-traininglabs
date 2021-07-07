package ai.certifai.solution.Self;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;

//import javax.xml.validation.Schema;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
/***
 * 
 *
 *
 * */

/*  We would be using the Boston Housing Dataset for this regression exercise.
                 *  This dataset is obtained from https://www.kaggle.com/vikrishnan/boston-house-prices
                 *  This dataset consist of 13 features and 1 label, the description are as follow:
                 *
                 *  CRIM: Per capita crime rate by town
                 *  ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
                 *  INDUS: Proportion of non-retail business acres per town
                 *  CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                 *  NOX: Nitric oxide concentration (parts per 10 million)
                 *  RM: Average number of rooms per dwelling
                 *  AGE: Proportion of owner-occupied units built prior to 1940
                 *  DIS: Weighted distances to five Boston employment centers
                 *  RAD: Index of accessibility to radial highways
                 *  TAX: Full-value property tax rate per $10,000
                 *  PTRATIO: Pupil-teacher ratio by town
                 *  B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
                 *  LSTAT: Percentage of lower status of the population
                 *  MEDV: Median value of owner-occupied homes in $1000s (target variable)
                 */

public class BostonHousePricePred {
    private static int seed = 123;
    private static int batchsize = 54;
    private static double learningRate = 0.001;
    private static int numFeatures = 13;
    private static int epochs = 2000;
    public static void main(String[] args) throws IOException, InterruptedException {

        File file = new ClassPathResource("boston/bostonHousing.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);// Split root dir into file
        CSVRecordReader csvRecordReader = new CSVRecordReader();
        csvRecordReader.initialize(fileSplit);

        Schema inputSchema = new Schema.Builder()
                .addColumnsDouble("CRIM","ZN","INDUS")
                .addColumnCategorical("CHAS","0","1")
                .addColumnsDouble("NOX","RM","AGE","DIS")
                .addColumnInteger("RAD")
                .addColumnsDouble("TAX","PTRATIO","B","LSTAT","MEDV")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema).build();

        List<List<Writable>> originalData = new ArrayList<>();
        while(csvRecordReader.hasNext()){
            originalData.add(csvRecordReader.next());
        }

        List<List<Writable>> transfromedData = LocalTransformExecutor.execute(originalData,tp);

        CollectionRecordReader crr = new CollectionRecordReader(transfromedData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr, transfromedData.size(),13,13,true);

        DataSet fullData = dataIter.next();
        fullData.shuffle();

        SplitTestAndTrain splitTestAndTrain = fullData.splitTestAndTrain(0.7);
        DataSet testingData = splitTestAndTrain.getTest();
        DataSet trainingData = splitTestAndTrain.getTrain();

        ViewIterator trainIter = new ViewIterator(trainingData,batchsize);
        ViewIterator testIter = new ViewIterator(testingData,batchsize);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .seed(seed)
                .l2(0.001)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numFeatures).nOut(256).activation(Activation.RELU)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nIn(256).nOut(128).activation(Activation.RELU)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nIn(128).nOut(64).activation(Activation.RELU)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .nIn(64).nOut(1).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(1000));
        model.fit(trainIter,epochs);

        //Model Evaluation
        RegressionEvaluation RegEval = model.evaluateRegression(testIter);
        System.out.println(RegEval.stats());

    }
}
