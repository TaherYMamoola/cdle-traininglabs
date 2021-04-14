package ai.certifai.training.classification.MessyVClean2;

import org.datavec.image.transform.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
//import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MessyVCleanEditLastLayerOthersFrozen {
    private static int seed = 123;
    private static final Random rand = new Random(seed);
    private static int height =224; // must follow the Transfer Learning
    private static int width =224;
    private static int nChannels =3;
    private static int batchSize = 54;
    private static int nClasses =2;

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

        DataSetIterator trainData = iterator.getTrain(transform);
        DataSetIterator testData = iterator.getTest();

        ZooModel zooModel = VGG16.builder().build();


    }
}
