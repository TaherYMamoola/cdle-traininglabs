package ai.certifai.solution.Self.MedicalCostPred;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.dataset.ViewIterator;

public class Evaluation {
    public static void eval(MultiLayerNetwork model, ViewIterator trainData, ViewIterator testData){
        RegressionEvaluation trainregEval = model.evaluateRegression(trainData);
        System.out.println(trainregEval.stats());
        RegressionEvaluation testregEval = model.evaluateRegression(testData);
        System.out.println(testregEval.stats());
    }
}
