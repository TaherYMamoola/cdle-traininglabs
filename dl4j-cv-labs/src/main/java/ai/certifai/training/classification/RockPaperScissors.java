package ai.certifai.training.classification;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.RandomCropTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class RockPaperScissors {

    private static String [] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static PathLabelGenerator labelmaker = new ParentPathLabelGenerator();
    private static Random rand = new Random();
    private static double splitRatio = 0.8;
    private static int height = 80;
    private static int width = 80;
    private static int nChannels = 3;
    private static int nClasses = 3;
    private static int batch_size = 54;
    private static int seed = 123;
    private static double learning_rate = 0.0001;
    private static double l2 = 0.0001;
    private static int epochs = 100;



    public static void main(String[] args) throws IOException {
        File InputFile = new ClassPathResource("rock_paper_scissors").getFile();
        FileSplit fileSplit = new FileSplit(InputFile,allowedExt);

        //Split Images
        PathFilter pathFilter = new BalancedPathFilter(rand,allowedExt,labelmaker);
        InputSplit [] allData = fileSplit.sample(pathFilter,splitRatio,1-splitRatio);
        InputSplit trainData = allData[0];
        InputSplit testnData = allData[1];

        //Read Images
        ImageRecordReader trainrr = new ImageRecordReader(height,width,nChannels,labelmaker);
        ImageRecordReader testrr = new ImageRecordReader(height,width,nChannels,labelmaker);
        trainrr.initialize(trainData);
        testrr.initialize(testnData);

        //Iterator
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainrr,batch_size,1,nClasses);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testrr,batch_size,1,nClasses);

        //Feature Scalling
        DataNormalization scaler = new ImagePreProcessingScaler();
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        //NN Architecture
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learning_rate))
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nIn(nChannels)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(20)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .setInputType(InputType.convolutional(height,width,nChannels))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        model.fit(trainIter,epochs);

        //Evaluation
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);
        System.out.println("Train Eval : "+ evalTrain.stats());
        System.out.println("Test Eval : "+ evalTest.stats());

//        File location = new File()
    }
}
