package dk.hapshapshaps.machinelearning.objectdetection;

import dk.hapshapshaps.machinelearning.objectdetection.models.Recognition;

import java.awt.image.BufferedImage;
import java.util.ArrayList;

public interface ObjectDetector extends AutoCloseable {
    ArrayList<Recognition> classifyImage(BufferedImage image);
}
