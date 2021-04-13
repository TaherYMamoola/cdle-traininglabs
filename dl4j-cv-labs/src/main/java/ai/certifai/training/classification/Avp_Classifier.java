package ai.certifai.training.classification;

//import javafx.util.Pair;
import jdk.internal.util.xml.impl.Input;
import org.bytedeco.javacv.ImageTransformer;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;



import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Avp_Classifier {

    private static String [] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static double splitRatio = 0.8;
    private static Random rand = new Random();
    private static PathLabelGenerator lableMaker = new ParentPathLabelGenerator();
    private static int height = 80;
    private static int width = 80;
    private static int nChannels = 3;
    private static int batch_size = 20;
    private static double learning_rate = 0.001;
    private static int epoch = 50;
    private static int seed = 123;
    private static double l2 = 0.0001;
    private static int nClasses = 2;

    public static void main(String[] args) throws IOException, InterruptedException {

        File inputFile = new ClassPathResource("Avp").getFile();
        FileSplit fileSplit = new FileSplit(inputFile,allowedExt); // to get the actual file

        //Split Traing & Test
        PathFilter pathFilter = new BalancedPathFilter(rand,allowedExt,lableMaker);

        InputSplit [] allData = fileSplit.sample(pathFilter,splitRatio,1-splitRatio);
        InputSplit trainData = allData[0];// position depends on the line 32 splitRatio location with 1-splitRatio location
        InputSplit testData = allData[1];

        //Image Transformation (Augmentation)
        ImageTransform hflip = new FlipImageTransform(1);
        ImageTransform rotate = new RotateImageTransform(15);
        RandomCropTransform rCrop = new RandomCropTransform(60,60);
//        ImageTransform show = new ShowImageTransform("Augmentation");

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(hflip,0.2),
                new Pair<>(rotate,0.3),
//                new Pair<>(show,1.0),
                new Pair<>(rCrop,0.2)
        );

        PipelineImageTransform transform = new PipelineImageTransform((java.util.List<Pair<ImageTransform, Double>>) pipeline,false);

        // Read Images
        ImageRecordReader trainrr = new ImageRecordReader(height,width,nChannels,lableMaker);
        ImageRecordReader testrr = new ImageRecordReader(height,width,nChannels,lableMaker);
        trainrr.initialize(trainData,transform);
//        trainrr.initialize(trainData);
        testrr.initialize(testData);

        //Iterator
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainrr,batch_size,1,nClasses);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testrr,batch_size,1,nClasses);

        DataNormalization scaler = new ImagePreProcessingScaler(); // makes images from 0-1 , douuble a n b are range of values
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learning_rate))
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .nIn(nChannels)
                        .nOut(16)// #filters
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(PoolingType.MAX)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .setInputType(InputType.convolutional(height,width,nChannels)) // avoid writing nIn() at every layer with this line
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        model.fit(trainIter,epoch);
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest= model.evaluate(testIter);

        System.out.println("Train Eval : "+evalTrain.stats());
        System.out.println("Test Eval : "+evalTest.stats());
//        Evaluation evaluate = model.evaluate(evalTrain);
//        for(int i =0 ;i<epoch;i++){
//            System.out.println("Epochs : "+i+"Accuracy : "+evaluate.accuracy());
//        }









    }
}
