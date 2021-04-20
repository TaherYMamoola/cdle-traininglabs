package ai.certifai.training.classification.IrdologyClassification;

import ai.certifai.Helper;
import ai.certifai.training.classification.MessyVClean2.MessyVCleanEditLastLayerOthersFrozen;
import ai.certifai.utilities.Visualization;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

public class irdologyClassifier_CG {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MessyVCleanEditLastLayerOthersFrozen.class);

    private static int seed = 123;
    private static final Random rand = new Random(seed);
    private static int height =224; // must follow the Transfer Learning
    private static int width =224;
    private static int nChannels =1;
    private static int batchSize = 100;
    private static int nClasses =2;
    private static double learning_rate = 0.001;
    private static int epochs = 10;
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/Irdology_Classifier.zip");
    private static ComputationGraph model;
    private static FineTuneConfiguration finetuner;
    private static List<String> label;
    private static DataSetIterator testIter;

    public static void main(String[] args) throws IOException {

         /*
        Initialize image augmentation
        */
//        ImageTransform horizontalFlip = new FlipImageTransform(1);
//        ImageTransform cropImage = new CropImageTransform(2
//        5);
//        ImageTransform rotateImage = new RotateImageTransform(rand, 15);
//        ImageTransform showImage = new ShowImageTransform("Image",1000);
//        boolean shuffle = false;
//        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
//                new Pair<>(horizontalFlip,0.5),
//                new Pair<>(rotateImage, 0.5),
//                new Pair<>(cropImage,0.3)
//                //                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
//        );
//
//        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        File InputFile = new ClassPathResource("IrdologyCholesterolClassification").getFile();

        IrdologyIterator iterator = new IrdologyIterator();
        iterator.setup(InputFile, height, width, nChannels, batchSize, nClasses);

        DataSetIterator trainIter = iterator.getTrain();
        testIter = iterator.getTest();
        label = trainIter.getLabels();

        //        If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
        if (modelFilename.exists()) {

            //        STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);

        } else {
            Nd4j.getRandom().setSeed(seed);
//            INDArray priors = Nd4j.create(priorBoxes);

            //     STEP 2 : Train the model using Transfer Learning
            //     STEP 2.1: Transfer Learning steps - Load TinyYOLO prebuilt model.
            log.info("Build model...");
            ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained();

            //     STEP 2.2: Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            //     STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            model = getComputationGraph(pretrained, fineTuneConf);
            System.out.println(model.summary(InputType.convolutional(height, width, nClasses)));

            //     STEP 2.4: Training and Save model.
            log.info("Train model...");
//            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
//            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 1; i < epochs + 1; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
        }
        //     STEP 3: Evaluate the model's accuracy by using the test iterator.
        evaluate();

        

    }
    
    
    private static ComputationGraph getComputationGraph(ComputationGraph vgg16, FineTuneConfiguration fineTuneConf){
        return new TransferLearning.GraphBuilder(vgg16)
                .nInReplace("block1_conv1",1,WeightInit.XAVIER)
                .fineTuneConfiguration(fineTuneConf)
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

    }

    

    private static FineTuneConfiguration getFineTuneConfiguration() {
    return new FineTuneConfiguration.Builder()
                .updater(new Adam(learning_rate))
                .seed(seed)
                .build();
    }
    
    
    private static void evaluate () throws IOException {
        Evaluation eval = new Evaluation(2);
        
        // EXPORT IMAGES
        File exportDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.export-images")
        ).toFile();

        if (!exportDir.exists()) {
            exportDir.mkdir();
        }

        float IOUTotal = 0;
        int count = 0;
        while (testIter.hasNext()) {
            DataSet imageSetVal = testIter.next();

            INDArray predictVal = model.output(imageSetVal.getFeatures())[0];
            INDArray labels = imageSetVal.getLabels();

//            // Uncomment the following if there's a need to export images
//            if (count % 5 == 0) {
//                Visualization.export(exportDir, imageSetVal.getFeatures(), imageSetVal.getLabels(), predictVal, count);
//            }

            count++;

            eval.eval(labels, predictVal);


            //STEP 5: Complete the code for IOU calculation here
            //Intersection over Union:  TP / (TP + FN + FP)
            float IOUCholestol = (float) eval.truePositives().get(1) / ((float) eval.truePositives().get(1) + (float) eval.falsePositives().get(1) + (float) eval.falseNegatives().get(1));
            IOUTotal = IOUTotal + IOUCholestol;

            System.out.println("IOU Cholestrol_Ring " + String.format("%.3f", IOUCholestol));

            eval.reset();

//            Visualization.visualize(
//                    imageSetVal.getFeatures(),
//                    imageSetVal.getLabels(),
//                    predictVal,
//                    frame,
//                    panel,
//                    batchSize,
//                    height,
//                    width
//            );

        }

        System.out.println("Mean IOU: " + IOUTotal / count);
    }

}
