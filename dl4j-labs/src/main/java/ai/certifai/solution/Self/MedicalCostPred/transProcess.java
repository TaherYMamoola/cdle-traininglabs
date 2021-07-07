package ai.certifai.solution.Self.MedicalCostPred;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;

public class transProcess {
    public static TransformProcess tp(Schema schema){
        TransformProcess tps = new TransformProcess.Builder(schema)
                .categoricalToInteger("region")
                .categoricalToOneHot("sex","smoker")
                .build();
    return tps;
    }
}
