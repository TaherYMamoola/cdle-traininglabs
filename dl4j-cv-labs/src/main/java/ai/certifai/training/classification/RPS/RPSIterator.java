package ai.certifai.training.classification.RPS;

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

public class RPSIterator {
    private static String [] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static InputSplit trainData, testData;
    private static Random rand = new Random();
    private static PathLabelGenerator labelmaker = new ParentPathLabelGenerator();
    private static PathFilter pathFilter = new BalancedPathFilter(rand,allowedExt,labelmaker);
    private static double splitRatio = 0.8;
    private static int height;
    private static int width;
    private static int nChannels;
    private static ImageTransform imgTransform;
    private static int batch_size;
    private static int nClasses;


    public void RPSIterator(){

    }

    public void setup(File inputFile,int Height, int Width, int numChannels, int batchSize, int numClasses){
        height = Height;
        width = Width;
        nChannels = numChannels;
        batch_size = batchSize;
        nClasses = numClasses;

        FileSplit split = new FileSplit(inputFile,allowedExt);
        InputSplit [] allData = split.sample(pathFilter, splitRatio, 1-splitRatio );
        trainData = allData[0];
        testData = allData[1];
    }

    public DataSetIterator getTrain (ImageTransform imageTransform) throws IOException {
        imgTransform = imageTransform;
        return makeIterator(true);

    }

    public DataSetIterator getTest () throws IOException {
        return makeIterator(false);

    }

    private DataSetIterator makeIterator (boolean train) throws IOException {
        ImageRecordReader rr = new ImageRecordReader(height,width,nChannels,labelmaker);

        if (train){
            rr.initialize(trainData,imgTransform);
        }else{
            rr.initialize(testData);
        }

        DataSetIterator iter = new RecordReaderDataSetIterator(rr,batch_size,1,nClasses);

        DataNormalization scaler = new ImagePreProcessingScaler();
        iter.setPreProcessor(scaler);

        return iter;
    }
}
