import dk.hapshapshaps.machinelearning.classifier.CustomClassifier;
import dk.hapshapshaps.machinelearning.classifier.models.ClassifyRecognition;
import dk.hapshapshaps.machinelearning.objectdetection.CustomObjectDetector;
import dk.hapshapshaps.machinelearning.objectdetection.ObjectDetector;
import dk.hapshapshaps.machinelearning.objectdetection.models.Box;
import dk.hapshapshaps.machinelearning.objectdetection.models.ObjectRecognition;
import dk.hapshapshaps.machinelearning.objectdetection.models.RectFloats;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        File imageFile = new File("/home/jacob/andet/training/docker-training-shared/random-test-images/image-6.jpg");

        File resultImageFile = new File("/home/jacob/andet/training/docker-training-shared/random-test-images/result.jpg");

//        classification(imageFile, resultImageFile);

        objectDetection(imageFile, resultImageFile);
    }

    private static void classification(File imageFile, File resultImageFile) throws IOException {
        File modelFile = new File("/home/jacob/andet/training/docker-training-shared/classification/subs/trained-files/output_graph.pb");
        File labelFile = new File("/home/jacob/andet/training/docker-training-shared/classification/subs/trained-files/output_labels.txt");

        BufferedImage image = ImageIO.read(imageFile);

        CustomClassifier classifier = new CustomClassifier(modelFile, labelFile);

        ClassifyRecognition recognition = classifier.classifyImage(image);

        String s = "";
    }

    private static void objectDetection(File imageFile, File resultImageFile) throws IOException {
        File modelFile = new File("/home/jacob/andet/training/docker-training-shared/subTraining/trainingResults/frozen_inference_graph.pb");
        File labelFile = new File("/home/jacob/andet/training/docker-training-shared/subTraining/data/object-detection.pbtxt");

        BufferedImage image = ImageIO.read(imageFile);

        ObjectDetector objectDetector = new CustomObjectDetector(modelFile, labelFile);

        ArrayList<ObjectRecognition> objectRecognitions = objectDetector.classifyImage(image);

        List<Box> boxes = new ArrayList<>();
        for (ObjectRecognition objectRecognition : objectRecognitions) {
            if(objectRecognition.getConfidence() > 0.05f) {
                RectFloats location = objectRecognition.getLocation();
                int x = (int) location.getX();
                int y = (int) location.getY();
                int width = (int) location.getWidth() - x;
                int height = (int) location.getHeight() - y;

                boxes.add(new Box(x, y, width, height));
            }
        }

        BufferedImage boxedImage = drawBoxes(image, boxes);

        ImageIO.write(boxedImage, "jpg", resultImageFile);

        String s = "";
    }

    private static BufferedImage drawBoxes(BufferedImage image, List<Box> boxes) {
        Graphics2D graph = image.createGraphics();
        graph.setColor(Color.green);

        for (Box box : boxes) {
            graph.drawRect(box.x, box.y, box.width, box.height);
        }

        graph.dispose();
        return image;
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
