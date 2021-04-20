package ai.certifai.training.classification.IrdologyClassification;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class grayscalling {

    public static void main(String[] args) throws IOException {
        try {
            final File dir = new File("C:\\Users\\Admin\\Desktop\\CertifAI\\GroupProject\\Project DataSet\\Cholestrol_JPG-20210417T044638Z-001\\Cholestrol_JPG");
            for(final File imgFile : dir.listFiles()){
                System.out.println(imgFile.toString());

//            File input = new File("...");
                BufferedImage image = ImageIO.read(imgFile);

                BufferedImage result = new BufferedImage(
                        image.getWidth(),
                        image.getHeight(),
                        BufferedImage.TYPE_INT_RGB);

                Graphics2D graphic = result.createGraphics();
                graphic.drawImage(image, 0, 0, Color.WHITE, null);

                for (int i = 0; i < result.getHeight(); i++) {
                    for (int j = 0; j < result.getWidth(); j++) {
                        Color c = new Color(result.getRGB(j, i));
                        int red = (int) (c.getRed() * 0.299);
                        int green = (int) (c.getGreen() * 0.587);
                        int blue = (int) (c.getBlue() * 0.114);
                        Color newColor = new Color(
                                red + green + blue,
                                red + green + blue,
                                red + green + blue);
                        result.setRGB(j, i, newColor.getRGB());
                    }
                }

//            File output = new File(imgFile);
                ImageIO.write(result, "jpg", imgFile);
            }

        }  catch (IOException e) {
            e.printStackTrace();
        }

    }
}


