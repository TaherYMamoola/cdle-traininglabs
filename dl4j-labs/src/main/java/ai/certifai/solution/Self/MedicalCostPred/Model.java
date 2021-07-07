package ai.certifai.solution.Self.MedicalCostPred;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;

public class Model {
    private static int epochs = 100;
    public static MultiLayerNetwork model(MultiLayerConfiguration config, ViewIterator trainData){
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        model.fit(trainData,epochs);
    return model;
    }
}
