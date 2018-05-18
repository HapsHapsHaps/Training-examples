package org.tensorflow.demo.custom;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class CustomClassifier implements AutoCloseable {

    String[] resultOperationNames = new String[] {"num_detections", "detection_boxes", "detection_scores",
            "detection_classes", "detection_masks"};
    private static final int MAX_RESULTS = 100;
    private static final int NUM_DETECTIONS = 2;
    private static final String IMAGE_FEED_NAME = "image_tensor";

    private Session session;
    private Session.Runner runner;
    private List<Tensor<?>> outputTensors = new ArrayList<>(5);
    private List<Tensor<?>> inputTensors = new ArrayList<>(1);
    private List<String> operationNames = new ArrayList<>(resultOperationNames.length);

    private float[] num_detections = new float[NUM_DETECTIONS];
    private float[] detection_boxes = new float[MAX_RESULTS * 4];
    private float[] detection_scores = new float[MAX_RESULTS];
    private float[] detection_classes = new float[MAX_RESULTS];
    private float[] detection_masks = new float[MAX_RESULTS];

    public CustomClassifier(Graph graph) {
        this.session = new Session(graph);
        this.runner = session.runner();
    }

    public void feedImage(Tensor<Float> tensorImage) {
        runner.feed(IMAGE_FEED_NAME, tensorImage);
        inputTensors.add(tensorImage);
    }

    public void run() {

        for (String operationName : resultOperationNames) {
            operationNames.add(operationName);
            runner.fetch(operationName);
        }

        outputTensors = runner.run();

        for (Tensor<?> tensor : inputTensors) {
            tensor.close();
        }
    }

    public float[] get_num_detections() {
        getTensorFloat("num_detections", num_detections);
        return num_detections;
    }

    public float[] get_detection_boxes() {
        getTensorFloat("detection_boxes", detection_boxes);
        return detection_boxes;
    }

    public float[] get_detection_scores() {
        getTensorFloat("detection_scores", detection_scores);
        return detection_scores;
    }

    public float[] get_detection_classes() {
        getTensorFloat("detection_classes", detection_classes);
        return detection_classes;
    }

    public float[] get_detection_masks() {
        getTensorFloat("detection_masks", detection_masks);
        return detection_masks;
    }

    private void getTensorFloat(String name, float[] output) {
        FloatBuffer buffer = FloatBuffer.wrap(output);
        getTensor(name).writeTo(buffer);
    }

    private Tensor<?> getTensor(String name) {
        for (int i = 0; i < operationNames.size() ; i++) {
            if(operationNames.get(i).equals(name)) {
                return outputTensors.get(i);
            }
        }
        return null;
    }

    @Override
    public void close() {
        for (Tensor<?> tensor : outputTensors) {
            tensor.close();
        }



        session.close();
    }
}
