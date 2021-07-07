package ai.certifai.solution.Self;

//import jdk.jfr.internal.WriteableUserPath;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BirdClass_Self {

    private static int seed = 123;
    private static int nFeatures = 10;
    private static int epochs = 1000;
    private static double splitRatio = 0.7;
    private static double learningRate = 0.001;
    private static double regularization = 0.0001;


    public static void main(String[] args) throws IOException, InterruptedException {

        File file = new ClassPathResource("birdclassify/bird.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);
        CSVRecordReader rr = new CSVRecordReader(1,",");
        rr.initialize(fileSplit);

        List<List<Writable>> OriData = new ArrayList<>();
        while(rr.hasNext()){
            OriData.add(rr.next());
        }

        Schema schema = new Schema.Builder()
                .addColumnInteger("id")
                .addColumnsDouble("huml","humw","ulnal","ulnaw","feml","femw","tibl","tibw","tarl","tarw")
                .addColumnCategorical("type","SW","W","T","R","P","SO")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("id")
                .categoricalToInteger("type")
                .build();

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(OriData,tp);

        CollectionRecordReader crr = new CollectionRecordReader(transformedData);
        DataSetIterator iter = new RecordReaderDataSetIterator(crr, transformedData.size(),-1,6);

        DataSet fullData = iter.next();
        fullData.shuffle(seed);

        SplitTestAndTrain split = fullData.splitTestAndTrain(splitRatio);
        DataSet trainingData  = split.getTrain();
        DataSet testData = split.getTest();

        DataNormalization norm = new NormalizerMinMaxScaler();
        norm.fit(trainingData);
        norm.transform(trainingData);
        norm.transform(testData);

        ViewIterator trainIter = new ViewIterator(trainingData, iter.batch());
        ViewIterator testIter = new ViewIterator(testData, iter.batch());

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate)).weightInit(WeightInit.XAVIER).seed(seed)//.l2(regularization)
                .list()
                .layer(0,new DenseLayer.Builder().nIn(nFeatures).nOut(256).activation(Activation.RELU).build())
                .layer(1,new DenseLayer.Builder().nIn(256).nOut(128).activation(Activation.RELU).build())
                .layer(2,new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
                .layer(3,new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
                .layer(4,new DenseLayer.Builder().nIn(32).nOut(16).activation(Activation.RELU).build())
                .layer(5,new OutputLayer.Builder().nIn(16).nOut(6).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        for(int i=0;i<=epochs;i++){
            model.fit(trainIter);
        }
        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);

        System.out.println("Train Eval: "+trainEval);
        System.out.println("Test Eval: "+testEval);




    }
}
