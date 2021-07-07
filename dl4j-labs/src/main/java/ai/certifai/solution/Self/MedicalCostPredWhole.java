package ai.certifai.solution.Self;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MedicalCostPredWhole {

    private static double splitRatio = 0.7;
    private static double regularization = 0.001;
    private static int seed = 123;
    private static int epochs = 40;
    private static int hidden = 1000;
    private static int batchSize = 1000;
    private static int nFeatures = 9;
    private static double learningRate = 0.001;


    public static void main(String[] args) throws IOException, InterruptedException {
//        Get File, Spilt path and file, get data from file
        File file = new ClassPathResource("medicalCost/insurance.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);
        CSVRecordReader rr = new CSVRecordReader(1,",");
        rr.initialize(fileSplit);
//          Schema shows all columns and values of categories
        Schema schema = new Schema.Builder()
                .addColumnInteger("age")
                .addColumnCategorical("sex", Arrays.asList("male","female"))
                .addColumnDouble("bmi")
                .addColumnInteger("children")
                .addColumnCategorical("smoker",Arrays.asList("yes","no"))
                .addColumnCategorical("region",Arrays.asList("northwest","southwest","northeast","southeast"))
                .addColumnsDouble("charges")
                .build();
//          Write into list<list<Writable>> format to transform data.
        List<List<Writable>> Oridata = new ArrayList<>();
        while(rr.hasNext()){
            Oridata.add(rr.next());
        }
//          TransformProcess to remove columns, transform categories to interger and oneHot encoding
        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("sex","smoker")
                .categoricalToOneHot("region")
                .build();

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(Oridata,tp);
//              CollectionRR to be transformed data from list,list,writable to iter
        CollectionRecordReader crr = new CollectionRecordReader(transformedData);
        DataSetIterator iter = new RecordReaderDataSetIterator(crr, transformedData.size(),7,7,true);

        DataSet fullData = iter.next();
        fullData.shuffle();
// Split data
        SplitTestAndTrain tnt = fullData.splitTestAndTrain(splitRatio);
        DataSet trainData = tnt.getTrain();
        DataSet testData = tnt.getTest();
// Normalize Dataset
        DataNormalization norm = new NormalizerMinMaxScaler();
        norm.fit(trainData);
        norm.transform(trainData);
        norm.transform(testData);
//ViewIterator to transform data from dataset to iter format
        ViewIterator trainIter = new ViewIterator(trainData,batchSize);
        ViewIterator testIter = new ViewIterator(testData,batchSize);
//Configure Model
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed).weightInit(WeightInit.XAVIER).updater(new Adam(learningRate)).l2(regularization)
                .list()
                .layer(0,new DenseLayer.Builder().nIn(nFeatures).nOut(hidden).activation(Activation.RELU).build())
                .layer(1,new DenseLayer.Builder().nOut(hidden).activation(Activation.RELU).build())
                .layer(2,new DenseLayer.Builder().nOut(hidden).activation(Activation.RELU).build())
                .layer(3,new DenseLayer.Builder().nOut(hidden).activation(Activation.RELU).build())
                .layer(4,new DenseLayer.Builder().nOut(hidden).activation(Activation.RELU).build())
                .layer(5,new OutputLayer.Builder().nIn(hidden).nOut(1).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR).build())
                .build();
//Init Model, set listerner, fit/train model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        model.fit(trainIter,epochs);
//Regression Eval
        RegressionEvaluation regEval = model.evaluateRegression(testIter);
        System.out.println(regEval.stats());




    }
}
