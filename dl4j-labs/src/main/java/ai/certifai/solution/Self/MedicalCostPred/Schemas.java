package ai.certifai.solution.Self.MedicalCostPred;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.schema.Schema;

import java.io.IOException;


public class Schemas {

    public static Schema Schemas1() throws IOException, InterruptedException {

        Schema sc = new Schema.Builder()
                .addColumnInteger("age")
                .addColumnCategorical("sex","female","male")
                .addColumnDouble("bmi")
                .addColumnInteger("children")
                .addColumnCategorical("smoker","yes","no")
                .addColumnCategorical("region","southwest","northwest","southeast","northeast")
                .addColumnDouble("charges")
                .build();
        return sc;
    }


}
