package ai.certifai.training.classification.MessyVClean2;

import ai.certifai.training.classification.transferlearning.EditLastLayerOthersFrozen;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;




public class MessyVCleanEditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MessyVCleanEditLastLayerOthersFrozen.class);

    private static int seed = 123;
    private static final Random rand = new Random(seed);
    private static int height =224; // must follow the Transfer Learning
    private static int width =224;
    private static int nChannels =3;
    private static int batchSize = 90;
    private static int nClasses =2;
    private static double learning_rate = 0.001;
    private static int epochs = 10;


    public static void main(String[] args) throws IOException {


         /*
        Initialize image augmentation
        */
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(rand, 15);
        ImageTransform showImage = new ShowImageTransform("Image",1000);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3)
    //                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );

        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        File InputFile = new ClassPathResource("MessyVClean1").getFile();

        MessyVCleanSetIterator iterator = new MessyVCleanSetIterator();
        iterator.setup(InputFile,height,width,nChannels,batchSize,nClasses);

        DataSetIterator trainIter = iterator.getTrain(transform);
        DataSetIterator testIter = iterator.getTest();

        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());


        FineTuneConfiguration finetuner = new FineTuneConfiguration.Builder()
                .updater(new Adam(learning_rate))
                .seed(seed)
                .build();

        ComputationGraph vgg16Transform = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(finetuner)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",new OutputLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .nIn(4096)
                        .nOut(nClasses)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build(),"fc2")
                .build();

        log.info(vgg16Transform.summary());
        StatsStorage statsStorage = new InMemoryStatsStorage();
        vgg16Transform.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(5),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        vgg16Transform.fit(trainIter,epochs);

    }
}
