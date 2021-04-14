package ai.certifai.training.classification.MessyVClean2;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class MessyVCleanSetIterator {

    private static String [] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static InputSplit trainData, testData;
    private static PathLabelGenerator lablemaker = new ParentPathLabelGenerator();
    private static Random rand = new Random();
    private static PathFilter pathFilter = new BalancedPathFilter(rand,allowedExt,lablemaker);
    private static int height;
    private static int width;
    private static int nChannels;
    private static ImageTransform imgTransform;
    private static int batchSize;
    private static int nClasses;
    private static double splitRatio = 0.8;

    public void setup(File TrainInput, int Height, int Width, int numChannels, int batch_size, int nClass){

        FileSplit fileSplits = new FileSplit(TrainInput,allowedExt);
        InputSplit[] allData = fileSplits.sample(pathFilter,splitRatio,1-splitRatio);
        trainData = allData[0];
        testData = allData[1];
        height = Height;
        width = Width;
        nChannels = numChannels;
        batchSize = batch_size;
        nClasses = nClass;



    }

    public DataSetIterator getTrain(ImageTransform imageTransform) throws IOException {
        imgTransform = imageTransform;
        return makeIter(true);
    }

    public DataSetIterator getTest() throws IOException {
        return makeIter(false);
    }

    private DataSetIterator makeIter (boolean train) throws IOException {

        ImageRecordReader rr = new ImageRecordReader(height,width, nChannels, lablemaker);

        if (train){
            rr.initialize(trainData,imgTransform);
        }else{
            rr.initialize(testData);
        }

        DataSetIterator iter = new RecordReaderDataSetIterator(rr,batchSize,1,nClasses);

        DataNormalization scaler = new ImagePreProcessingScaler();
        iter.setPreProcessor(scaler);

        return iter;

    }
}
