package ai.certifai.training.WineClassification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

//import javax.xml.validation.Schema;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;



public class WineClassification {
    static int seed = 123;
    static int numberInput = 11;
    static int numberClasses = 6;
    static double learning_rate = 0.001;
    static int epochs = 100;

    public static void main(String[] args) throws IOException, InterruptedException {

        File src = new ClassPathResource("winequality-red.csv").getFile();

        FileSplit fileSplit = new FileSplit(src);

        RecordReader rr = new CSVRecordReader(1,',');// since there is header in the file remove row 1 with delimter coma
        rr.initialize(fileSplit);

        Schema sc = new Schema.Builder()
                .addColumnsDouble("fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol")
                .addColumnCategorical("quality", Arrays.asList("3","4","5","6","7","8"))
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .categoricalToInteger("quality")
                .build();

        System.out.println("Initial Schema : "+tp.getInitialSchema());
        System.out.println("New Schema : "+tp.getFinalSchema());

        List<List<Writable>> original_data = new ArrayList<>();

        while(rr.hasNext()){
            original_data.add(rr.next());
        }

        List<List<Writable>> transformed_data = LocalTransformExecutor.execute(original_data,tp);

        CollectionRecordReader crr = new CollectionRecordReader(transformed_data);

        DataSetIterator iter = new RecordReaderDataSetIterator(crr,transformed_data.size(),-1,6);

        DataSet fullDataSet = iter.next();
        fullDataSet.shuffle(seed);

        SplitTestAndTrain traintestsplit = fullDataSet.splitTestAndTrain(0.8);
        DataSet trainDataSet = traintestsplit.getTrain();
        DataSet testDataSet = traintestsplit.getTest();


        DataNormalization normalization = new NormalizerMinMaxScaler();
        normalization.fit(trainDataSet);
        normalization.transform(trainDataSet);
        normalization.transform(testDataSet);

        MultiLayerConfiguration config = getConfig (numberInput,numberClasses,learning_rate);

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        //UI-Evaluator
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //Set model listeners
        model.setListeners(new StatsListener(storage, 10));

        Evaluation eval;
        model.fit(trainDataSet);
        for(int i=0;i<=epochs;i++) {
            eval = model.evaluate(new ViewIterator(testDataSet, transformed_data.size()));
            System.out.println("Epoch " + i + ", Accuracy : "+eval.accuracy());
        }

        Evaluation evalTrain = model.evaluate(new ViewIterator(trainDataSet,transformed_data.size()));
        Evaluation evalTest = model.evaluate(new ViewIterator(testDataSet,transformed_data.size()));

        System.out.println("Train Eval : "+evalTrain.stats());
        System.out.println("Test Eval : "+evalTest.stats());







    }

    public static MultiLayerConfiguration getConfig (int numberInput, int numberClasses, double learning_rate){

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learning_rate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numberInput)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .nOut(numberClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.POISSON)
                        .build())
                .build();
        return config;
    }
}
