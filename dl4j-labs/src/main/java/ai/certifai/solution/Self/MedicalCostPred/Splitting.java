package ai.certifai.solution.Self.MedicalCostPred;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

public class Splitting {
    private static double split = 0.7;

    public static DataSet[] tnt1(DataSet dataSet){
        SplitTestAndTrain tnt = dataSet.splitTestAndTrain(split);
        DataSet[] arr = new DataSet[2];
        arr[0] = tnt.getTrain();
        arr[1] = tnt.getTest();
        return arr;

    }
}
