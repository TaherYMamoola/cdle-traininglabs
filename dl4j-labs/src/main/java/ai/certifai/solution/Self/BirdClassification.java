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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
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

public class BirdClassification {
    private static int seed = 123;
    private static int nFeatures = 10;
    private static int nClass = 6;
    private static int epochs = 100;
    private static double splitRatio = 0.8;
    private static double lr = 0.0004;



    public static void main(String[] args) throws IOException, InterruptedException {
        File file = new ClassPathResource("birdclassify/bird.csv").getFile();
        FileSplit split = new FileSplit(file);

        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(split);

        List<List<Writable>> oriData = new ArrayList<>();
        while(rr.hasNext()){
            oriData.add(rr.next());
        }

        Schema sc = new Schema.Builder()
                .addColumnInteger("id")
                .addColumnsDouble("huml,humw,ulnal,ulnaw,feml,femw,tibl,tibw,tarl,tarw")
                .addColumnCategorical("type", Arrays.asList("P","R","SO","SW","T","W"))
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .removeColumns("id")
                .categoricalToInteger("type")
                .build();

        List<List<Writable>> processedData = LocalTransformExecutor.execute(oriData,tp);

        CollectionRecordReader crr = new CollectionRecordReader(processedData);
        DataSetIterator iter = new RecordReaderDataSetIterator(crr,processedData.size(),-1,6);

        DataSet fullData = iter.next();
        fullData.shuffle(seed);

        SplitTestAndTrain tNt = fullData.splitTestAndTrain(splitRatio);
        DataSet trainData = tNt.getTrain();
        DataSet testData = tNt.getTest();

        DataNormalization norm = new NormalizerMinMaxScaler();
        norm.fit(trainData);
        norm.transform(trainData);
        norm.transform(testData);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .l2(0.0001)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(nFeatures)
                        .nOut(8)//nFeatures+nClass)/2
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nIn(8)
                        .nOut(7)
                        .activation(Activation.RELU)
                        .build())
                .layer(2,new OutputLayer.Builder()
                        .nIn(7)
                        .nOut(nClass)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        model.fit((DataSetIterator) trainData,epochs);

        Evaluation TrainEval = model.evaluate((DataSetIterator) trainData);
        Evaluation TestEval = model.evaluate((DataSetIterator) testData);

        System.out.println("Train Eval ::"+TrainEval);
        System.out.println("Test Eval ::"+TestEval);


    }
}


