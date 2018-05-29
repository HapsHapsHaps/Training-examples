package dk.hapshapshaps.classifier.objectdetection;

import dk.hapshapshaps.classifier.objectdetection.models.Recognition;

import java.awt.image.BufferedImage;
import java.util.ArrayList;

public interface ObjectDetector {
    ArrayList<Recognition> classifyImage(BufferedImage image);
}
