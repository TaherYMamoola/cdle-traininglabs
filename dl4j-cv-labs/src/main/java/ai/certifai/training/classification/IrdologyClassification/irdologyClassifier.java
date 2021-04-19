package ai.certifai.training.classification.IrdologyClassification;

import ai.certifai.solution.object_detection.AvocadoBananaDetector.FruitDataSetIterator;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

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
    private static Frame frame = null;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar YELLOW = RGB(255, 255, 0);
    private static Scalar[] colormap = {GREEN, YELLOW}; // bbox color as per prediction output ; differentiate classes based on the pred
    private static String labeltext = null;
    private static ComputationGraph model;

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
