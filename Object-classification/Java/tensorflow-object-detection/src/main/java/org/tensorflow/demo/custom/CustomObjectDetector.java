package org.tensorflow.demo.custom;

import org.apache.commons.io.FileUtils;
import org.tensorflow.*;
import org.tensorflow.demo.GraphBuilder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;

public class CustomObjectDetector {

    final static int SIZE = 200;
    final static float MEAN = 255f;

    private byte[] graphBytes;

    public void addGraph(File graphFile) throws IOException {
        this.graphBytes = FileUtils.readFileToByteArray(graphFile);
    }

    public void classifyImage(BufferedImage image) {
        ByteArrayOutputStream imageStream = new ByteArrayOutputStream();
        try {
            ImageIO.write(image, "jpg", imageStream);
        } catch (IOException e) {
            throw new RuntimeException("Bad image conversion to byteArray.");
        }

        Tensor<Float> imageTensor = normalizeImage(imageStream.toByteArray());

        executeGraph(imageTensor);
    }

    /**
     * Pre-process input. It resize the image and normalize its pixels
     * @param imageBytes Input image
     * @return Tensor<Float> with shape [1][416][416][3]
     */
    public static Tensor<Float> normalizeImage(final byte[] imageBytes) {
        try (Graph graph = new Graph()) {
            GraphBuilder graphBuilder = new GraphBuilder(graph);

            final Output<Float> output =
                    graphBuilder.div( // Divide each pixels with the MEAN
                            graphBuilder.resizeBilinear( // Resize using bilinear interpolation
                                    graphBuilder.expandDims( // Increase the output tensors dimension
                                            graphBuilder.cast( // Cast the output to Float
                                                    graphBuilder.decodeJpeg(
                                                            graphBuilder.constant("input", imageBytes), 3),
                                                    Float.class),
                                            graphBuilder.constant("make_batch", 0)),
                                    graphBuilder.constant("size", new int[]{SIZE, SIZE})),
                            graphBuilder.constant("scale", MEAN));

            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
            }
        }
    }

    /**
     * Executes graph on the given preprocessed image
     * @param image preprocessed image
     * @return output tensor returned by tensorFlow
     */
    private float[] executeGraph(final Tensor<Float> image) {
        try (Graph graph = loadGraph()) {

            try(CustomClassifier classifier = new CustomClassifier(graph)) {
                classifier.feedImage(image);
                classifier.run();

                float[] num_detections = classifier.get_num_detections();
                float[] detection_boxes = classifier.get_detection_boxes();

                String s = "";

                String n = "j";
            }



        }
        return null;
    }

    private Graph loadGraph() {
        Graph graph = new Graph();
        graph.importGraphDef(graphBytes);
        return graph;
    }

//    private String[] fetchNames() {
//
//    }
}
