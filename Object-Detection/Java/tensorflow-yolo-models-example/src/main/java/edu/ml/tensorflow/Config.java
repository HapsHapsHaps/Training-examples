package edu.ml.tensorflow;

/**
 * Configuration file for TensorFlow Java Yolo application
 */
public interface Config {
//    String GRAPH_FILE = "/YOLO/yolo-voc.pb";
    String GRAPH_FILE = "/tensor/frozen_inference_graph.pb";
//    String LABEL_FILE = "/YOLO/yolo-voc-labels.txt";
    String LABEL_FILE = "/tensor/object-detection.pbtxt";

    // Params used for image processing
    int SIZE = 416;
    float MEAN = 255f;

    // Output directory
    String OUTPUT_DIR = "./sample";
}
