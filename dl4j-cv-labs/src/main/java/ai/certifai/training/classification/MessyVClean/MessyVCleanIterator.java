package ai.certifai.training.classification.MessyVClean;

import com.fasterxml.jackson.databind.ser.Serializers;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;

import java.io.File;
import java.util.Random;

public class MessyVCleanIterator {
    private static String [] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static InputSplit trainData, testData;
    private static PathLabelGenerator lablemaker = new ParentPathLabelGenerator();
    private static Random rand = new Random();
    private static PathFilter pathFilter = new BalancedPathFilter(rand,allowedExt,lablemaker);



    public void MessyVCleanClassifier(){

    }

    public void setup(File inputFile,double trainPrecent){
        FileSplit split = new FileSplit(inputFile,allowedExt);
        InputSplit [] allData = split.sample(pathFilter, trainPrecent, 1-trainPrecent);
        trainData = allData[0];
        testData = allData[1];



    }

}


