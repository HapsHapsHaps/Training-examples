import org.tensorflow.demo.Classifier;
import org.tensorflow.demo.Recognition;
import org.tensorflow.demo.RectFloats;
import org.tensorflow.demo.TensorFlowObjectDetectionAPIModel;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {

        File modelFile = new File("/home/jacob/andet/training/docker-training-shared/raccoon_dataset-master/trainingresult/output_inference_graph_custom.pb/frozen_inference_graph.pb");
        File labelFile = new File("/home/jacob/andet/training/docker-training-shared/raccoon_dataset-master/data/object-detection.pbtxt");
        int inputSize = 200;

        Classifier classifier = TensorFlowObjectDetectionAPIModel.create(modelFile, labelFile, inputSize);

//        File imageFile = new File("/home/jacob/andet/training/docker-training-shared/raccoon_dataset-master/images/raccoon-14.jpg");
//        File imageFile = new File("/home/jacob/temp/raccoon-small.jpg");
        File imageFile = new File("/home/jacob/temp/flower-small.jpg");
//        File imageFile = new File("/home/jacob/Downloads/dog-small.jpg");

        File resultImageFile = new File("/home/jacob/temp/raccoon-result.jpg");

        BufferedImage image = ImageIO.read(imageFile);

        List<Recognition> recognitions = classifier.recognizeImage(image);

        Recognition firstRecognition = recognitions.get(0);

        RectFloats location = firstRecognition.getLocation();

        int x = (int) location.getLeft();
        int y = (int) location.getTop();
        int width = (int) location.getRight() - x;
        int height = (int) location.getBottom() - y;

        BufferedImage boxedImage = drawBox(image, x, y, width, height);

        ImageIO.write(boxedImage, "jpg", resultImageFile);

        String s = "";

    }

    private static BufferedImage drawBox(BufferedImage image, int x, int y, int width, int height) {
        Graphics2D graph = image.createGraphics();
        graph.setColor(Color.BLACK);
//        graph.fill(new Rectangle(x, y, width, height));
        graph.drawRect(x, y, width, height);
//        graph.setStroke();
        graph.dispose();
        return image;
    }
}
