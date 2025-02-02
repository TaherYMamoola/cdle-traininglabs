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

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 *
 * 1. Go to https://image.online-convert.com/, convert resources/image_processing/opencv.png into the following format:
 *       - .bmp
 *       - .jpg
 *       - .tiff
 *     Save them to the same resources/image_processing folder.
 *
 *  2. Use the .imread function to load each all the images in resources/image_processing,
 *       and display them using Display.display
 *
 *
 *  3. Print the following image attributes:
 *       - depth
 *       - number of channel
 *       - width
 *       - height
 *
 *  4. Repeat step 2 & 3, but this time load the images in grayscale
 *
 *  5. Resize file
 *
 *  6. Write resized file to disk
 *
 * */

public class LoadImages {
    public static void main(String[] args) throws IOException {

        String img = new ClassPathResource("image_processing/opencv.png").getFile().getAbsolutePath();
        Mat source = imread(img);// Mat object is a simple matrix format array
        Display.display(source,"Img");
        System.out.println("Array Height : "+source.arrayHeight());
        System.out.println("Array Width : "+source.arrayWidth());
        System.out.println("# Channels : "+source.arrayChannels());

        // Image Resizing
        Mat downsize = new Mat();
        Mat upsize_Linear = new Mat();
        Mat upsize_BiLinear = new Mat();
        Mat upsize_Cubic = new Mat();
        Mat upsized_Nearest = new Mat();


        resize(source,downsize,new Size(300,300));
        resize(downsize,upsize_Linear, new Size(1200,1478),0,0,INTER_LINEAR); // V, and V1 is the scale factor to upsize in  horizontal and vertical, 0 means do it with size
        resize(downsize,upsize_BiLinear,new Size(1200,1478),0,0,INTER_LINEAR_EXACT);
        resize(downsize,upsize_Cubic,new Size(1200,1478),0,0,INTER_CUBIC);
        resize(downsize,upsized_Nearest,new Size(1200,1478),0,0,INTER_NEAREST);


        Display.display(downsize,"Downsized");
        Display.display(upsize_Linear,"Linear_Upsized");
        Display.display(upsize_BiLinear,"Linear_Exact Upsized");
        Display.display(upsize_Cubic,"Cubic Upsized");
        Display.display(upsized_Nearest,"Nearest Upsized");

    }
}
