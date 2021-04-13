package ai.certifai.training.classification.MessyVClean;

import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class MessyVCleanClassifier {
    private static double splitRatio = 0.8;

    public static void main(String[] args) throws IOException {

        File inputFile = new ClassPathResource("MessyVClean").getFile();

        MessyVCleanIterator iterator = new MessyVCleanIterator();
        iterator.setup(inputFile,splitRatio);
    }
}
