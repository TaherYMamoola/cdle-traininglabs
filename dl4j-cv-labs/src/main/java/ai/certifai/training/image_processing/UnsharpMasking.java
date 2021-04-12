/*
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.certifai.training.image_processing;

/*
* TASKS:
* -------
* 1. Load any image from the Resources folder
* 2. Apply Unsharp Masking by following the steps shown in the Day 6 lecture slides on Unsharp Masking
* 3. Display the follwing:
*       - the input image
*       - the "detail" (residual after removing smoothed image from the input)
*       - the sharpened image
*
* */

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_core.add;
import static org.bytedeco.opencv.global.opencv_core.subtract;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;

public class UnsharpMasking {
    public static void main(String[] args) throws IOException {

        Mat src = imread(new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath());

        Display.display(src,"Original Image");

        Mat blur = new Mat();

        GaussianBlur(src,blur, new Size(3,3),10);

        Mat details = new Mat();
        subtract(src,blur,details);

        Mat output = new Mat();
        add(src,details,output);

        Display.display(details,"Details");
        Display.display(output,"Result");


    }
}
