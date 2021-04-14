package ai.certifai.training.classification.MessyVClean;

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

public class MessyVCleanClassifier {
    private static double splitRatio = 0.8;
    private static int height = 80;
    private static int width = 80;
    private static int nChannels = 3;
    private static int batchSize = 54;
    private static int nClasses = 2;
    private static int seed = 123;
    private static int epochs = 1;
    private static double learning_rate = 0.001;




    public static void main(String[] args) throws IOException {

        File FileInput = new ClassPathResource("MessyVClean1").getFile();


        MessyVCleanIterator iterator = new MessyVCleanIterator();
        iterator.setup(FileInput,height, width,nChannels,batchSize,nClasses);

        ImageTransform rotate = new RotateImageTransform(15);
        ImageTransform rCrop = new RandomCropTransform(50,50);
        ImageTransform crop = new CropImageTransform(10);

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(rotate,0.2),
                new Pair<>(rCrop,0.3),
                new Pair<>(crop,0.2)
        );

        PipelineImageTransform transform = new PipelineImageTransform(pipeline,false);

        DataSetIterator trainIter = iterator.getTrain(transform);
        DataSetIterator testIter = iterator.getTest();

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learning_rate))
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .nIn(nChannels)
                        .activation(Activation.RELU)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(16)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nOut(25)
                        .activation(Activation.RELU)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .nOut(nClasses)
                        .lossFunction(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutional(height,width,nChannels))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.fit(trainIter,epochs);
        model.setListeners(new ScoreIterationListener(10));

        Evaluation TrainEval = model.evaluate(trainIter);
        Evaluation TestEval = model.evaluate(testIter);
        System.out.println("Train Eval : "+TrainEval.stats());
        System.out.println("Test Eval : "+TestEval.stats());


    }
}
