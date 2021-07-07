package ai.certifai.solution.Self.MedicalCostPred;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.jetbrains.annotations.NotNull;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class GetFile {

//    @org.jetbrains.annotations.Unmodifiable
    @NotNull
    public static FileSplit getFile() throws IOException, InterruptedException {
        File file  = new ClassPathResource("medicalCost/insurance.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);
        return fileSplit;
    }

    public GetFile(){

    }

}
