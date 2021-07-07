package ai.certifai.solution.Self.MedicalCostPred;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainFile {
    private static int batchSize = 50;
    public static void main(String[] args) throws IOException, InterruptedException {
        FileSplit fileSplit = GetFile.getFile();
        CSVRecordReader rr = new CSVRecordReader(1,",");
        rr.initialize(fileSplit);

        List<List<Writable>> Oridata = new ArrayList<>();
        while(rr.hasNext()){
            Oridata.add(rr.next());
        }

        Schema sc = Schemas.Schemas1();
        TransformProcess tp = transProcess.tp(sc);
        List<List<Writable>> transformedData = LocalTransformExecutor.execute(Oridata,tp);

        CollectionRecordReader crr = new CollectionRecordReader(transformedData);
        DataSetIterator iter = new RecordReaderDataSetIterator(crr, transformedData.size(),7,7,true);
        DataSet fulldata = iter.next();
        fulldata.shuffle();

        DataSet[] ds = Splitting.tnt1(fulldata);
        DataSet trainData = ds[0];
        DataSet testData = ds[1];

        DataNormalization norm = new NormalizerMinMaxScaler();
        norm.fit(trainData);
        norm.transform(trainData);
        norm.transform(testData);

        ViewIterator trainIter = new ViewIterator(trainData,batchSize);
        ViewIterator testIter = new ViewIterator(testData,batchSize);

        MultiLayerConfiguration config = ModelConfiguration.config();
        MultiLayerNetwork model = Model.model(config,trainIter);

        Evaluation.eval(model,trainIter,testIter);


    }
}
