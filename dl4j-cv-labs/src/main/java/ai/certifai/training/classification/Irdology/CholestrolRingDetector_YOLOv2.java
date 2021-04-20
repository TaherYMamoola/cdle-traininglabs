package ai.certifai.training.classification.Irdology;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;


public class CholestrolRingDetector_YOLOv2 {

    private static int batchSize = 15;
    private static List<String> labels;
    private static File modelFileName = new File(System.getProperty("user.dir"),"generated-models/CholestrolRingDetector_yolov2.zip");
    private static int seed = 123;
    private static final Logger log = LoggerFactory.getLogger(CholestrolRingDetector_YOLOv2.class);
    private static ComputationGraph model;
    private static double  priorboxes[][]={{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};
    private static double learningRate = 1e-4;
    private static int nBoxes = 5;
    private static int nClasses = 2;
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;
    private static int nEpochs = 1000;
    private static double detectionThreshold = 0.5;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar RED = RGB(255, 0, 0);
    private static Scalar[] colormap = {GREEN, RED};
    private static String labeltext = null;
    private static boolean trainingMode = true;

    public void CholestrolRingDetector_YOLOv2() throws IOException, InterruptedException {
        IrisDataSetIterator.setup();
        RecordReaderDataSetIterator testIter = IrisDataSetIterator.testIterator(1);
        offlineValidationWithTest(testIter);
    }


    public static void main(String[] args) throws Exception {
        // STEP 1: Set iterator
        IrisDataSetIterator.setup();
        RecordReaderDataSetIterator trainIter = IrisDataSetIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = IrisDataSetIterator.testIterator(1);

        labels = trainIter.getLabels();

        //        If model does not exist, create the model, else load the model
        if(modelFileName.exists()){
            //        STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("load model...");
            model = ModelSerializer.restoreComputationGraph(modelFileName);
        }
        else{
            Nd4j.getRandom().setSeed(seed);
            INDArray priors = Nd4j.create(priorboxes);
            //     STEP 2 : Train the model using Transfer Learning
            //     STEP 2.1: Transfer Learning steps - Load TinyYOLO prebuilt model.
            log.info("build model...");
            ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

            //     STEP 2.2: Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            //     STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            model = getComputationGraph(pretrained, priors, fineTuneConf);
        }

        System.out.println(model.summary(InputType.convolutional(
                IrisDataSetIterator.yoloHeight,
                IrisDataSetIterator.yoloWidth,
                nClasses)));

        if(trainingMode){
            //     STEP 2.4: Training and Save model.
            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1),new StatsListener(storage));

            for(int i = 1; i<nEpochs+1; i++){
                trainIter.reset();
                if(trainIter.hasNext()){
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model,modelFileName,true);
            System.out.print("Model saved.");
        }

        //     STEP 3: Evaluate the model's accuracy by using the test iterator.
        offlineValidationWithTest(testIter);


    }


    public static void offlineValidationWithTest(RecordReaderDataSetIterator iter) throws InterruptedException {
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();

        while (iter.hasNext() && canvas.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = iter.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            YoloUtils.nms(objs, 0.4);
            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));
            convertedMat_big = drawResults(objs, convertedMat_big, w, h);
            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        canvas.dispose();

    }

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / IrisDataSetIterator.gridW);
            int y1 = (int) Math.round(h * xy1[1] / IrisDataSetIterator.gridH);
            int x2 = (int) Math.round(w * xy2[0] / IrisDataSetIterator.gridW);
            int y2 = (int) Math.round(h * xy2[1] / IrisDataSetIterator.gridH);
            //Draw bounding box
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            //Display label text
            labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));
        }
        return mat;
    }

    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {
        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")
                .setFeatureExtractor("conv2d_22")
//                .nInReplace("conv2d_1", 1, WeightInit.XAVIER)
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()
                .seed(seed)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .l2(0.00001)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }
}
