package ai.certifai.training.classification.Irdology;


import ai.certifai.Helper;
import ai.certifai.utilities.VocLabelProvider;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class IrisDataSetIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(IrisDataSetIterator.class);
    private static String dataDir;
    private static Path trainDir, testDir;
    private static FileSplit trainData, testData;
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    public static final int yoloHeight = 416;
    public static final int yoloWidth = 416;
    private static final int nChannels = 3;
    public static final int gridH = 13;
    public static final int gridW = 13;

    public static void setup() throws IOException {
//        log.info("Load data...");
        loadData();
        // home path + datas path
//        trainDir = Paths.get(dataDir, "iris", "train");
//        testDir = Paths.get(dataDir,"iris", "test");
        File trainDir = new File((new ClassPathResource("train").getPath()));
        File testDir = new File((new ClassPathResource("test").getPath()));
        //grabbing the data
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()),NativeImageLoader.ALLOWED_FORMATS, rng);


    }

    private static void loadData() throws IOException {
        // home path
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws IOException {
        return makeIterator(trainData, trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws IOException {
        return makeIterator(testData, testDir, batchSize);
    }

    private static RecordReaderDataSetIterator makeIterator(FileSplit split, Path dir, int batchSize) throws IOException {
        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloHeight, yoloWidth, nChannels, gridH, gridW, new VocLabelProvider(dir.toString()));
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,1,true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0,1));

        return iter;
    }

}
