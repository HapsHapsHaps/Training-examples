package org.tensorflow.demo;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Generic interface for image recognition engines.
 */
public interface Classifier {

    List<Recognition> recognizeImage(BufferedImage bitmap);

    void enableStatLogging(final boolean debug);

    String getStatString();

    void close();
}
