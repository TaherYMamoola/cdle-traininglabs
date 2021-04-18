package ai.certifai.training.classification.IrdologyClassification;

import org.datavec.image.transform.*;
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
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class irdologyClassifier {

    private static int height = 80;
    private static int width = 80;
    private static int numChannels = 1;
    private static int batchSize = 54;
    private static int numClasses = 2;
    private static int seed = 123;
    private static int epoch = 10;
    private static double learningRate = 0.001;
    private static double l2 = 0.001;

    public static void main(String[] args) throws IOException {

        IrdologyIterator iterator = new IrdologyIterator();

        File inputFile = new ClassPathResource("IrdologyCholesterolClassification").getFile();

        iterator.setup(inputFile,height,width,numChannels,batchSize,numClasses);

        //Image Transformation
//        ImageTransform rotate = new RotateImageTransform(15);
//        ImageTransform rCrop = new RandomCropTransform(50,50);
//        ImageTransform crop = new CropImageTransform(5);
//        ImageTransform flip = new FlipImageTransform(0); // zero for horizontal flips, 1 for verticle
//
//        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
//                new Pair<>(rotate,0.2),
//                new Pair<>(rCrop,0.3),
//                new Pair<>(crop,0.2),
//                new Pair<>(flip,0.2)
//        );

//        PipelineImageTransform transform = new PipelineImageTransform(pipeline,false);

        DataSetIterator trainIter = iterator.getTrain();
        DataSetIterator testIter = iterator.getTest();

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
//                .l2(l2)
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nIn(numChannels)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(3,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.AVG)
                        .build())
                .layer(4,new DenseLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(5,new OutputLayer.Builder()
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .setInputType(InputType.convolutional(height,width,numChannels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        model.fit(trainIter,epoch);

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);
        System.out.println("Train Eval : "+evalTrain.stats());
        System.out.println("Test Eval : "+evalTest.stats());

//        File location = new File(System.getProperty("user.dir"),"generated_models/rockPaperScissors.zip");
//
//        ModelSerializer.writeModel(model,location,false);

    }
}
