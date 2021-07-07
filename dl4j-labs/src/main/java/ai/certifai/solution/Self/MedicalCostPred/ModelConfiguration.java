package ai.certifai.solution.Self.MedicalCostPred;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelConfiguration {
    private static int seed = 123;
    private static double learningRate = 0.001;
    private static int nFeatures = 9;
    public static MultiLayerConfiguration config(){
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed).weightInit(WeightInit.XAVIER).updater(new Adam(learningRate))
                .list()
                .layer(0,new DenseLayer.Builder().nIn(nFeatures).nOut(1200).activation(Activation.RELU).build())
                .layer(1,new DenseLayer.Builder().nIn(1200).nOut(600).activation(Activation.RELU).build())
                .layer(2,new DenseLayer.Builder().nIn(600).nOut(300).activation(Activation.RELU).build())
                .layer(3,new DenseLayer.Builder().nIn(300).nOut(150).activation(Activation.RELU).build())
                .layer(4,new DenseLayer.Builder().nIn(150).nOut(75).activation(Activation.RELU).build())
                .layer(5,new DenseLayer.Builder().nIn(75).nOut(35).activation(Activation.RELU).build())
                .layer(6,new DenseLayer.Builder().nIn(35).nOut(17).activation(Activation.RELU).build())
                .layer(7,new DenseLayer.Builder().nIn(17).nOut(8).activation(Activation.RELU).build())
                .layer(8,new DenseLayer.Builder().nIn(8).nOut(4).activation(Activation.RELU).build())
                .layer(9,new OutputLayer.Builder().nIn(4).nOut(1).activation(Activation.LEAKYRELU).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        return config;
    }

}
