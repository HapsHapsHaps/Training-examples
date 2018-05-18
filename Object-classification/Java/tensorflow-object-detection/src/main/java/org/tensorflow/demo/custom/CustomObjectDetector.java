package org.tensorflow.demo.custom;

import org.tensorflow.*;
import org.tensorflow.demo.GraphBuilder;
import org.tensorflow.demo.Recognition;
import org.tensorflow.demo.RectFloats;
import org.tensorflow.types.UInt8;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class CustomObjectDetector {

//    private static final int SIZE = 200;
    private static final float MEAN = 255f;
    private static final String INPUT_NAME = "image_tensor";
    private static final int MAX_RESULTS = 100;

    private byte[] graphBytes;
    private List<String> labels;

    public CustomObjectDetector(File graphFile, File labelFile) throws IOException {
        addGraph(graphFile);
        addLabels(labelFile);
    }

    private void addGraph(File graphFile) throws IOException {
//        this.graphBytes = FileUtils.readFileToByteArray(graphFile);

        InputStream inputStream = Files.newInputStream(graphFile.toPath());

        int baosInitSize = inputStream.available() > 16384 ? inputStream.available() : 16384;
        ByteArrayOutputStream baos = new ByteArrayOutputStream(baosInitSize);
        int numBytesRead;
        byte[] buf = new byte[16384];
        while ((numBytesRead = inputStream.read(buf, 0, buf.length)) != -1) {
            baos.write(buf, 0, numBytesRead);
        }
        this.graphBytes = baos.toByteArray();
    }

    private void addLabels(File labelFile) throws IOException {
        this.labels = new ArrayList(2);
        List<String> labels = Files.readAllLines(labelFile.toPath());

        for (String label : labels) {
            if(label.contains("name:")) {
                int i = label.indexOf("'");
                String substring = label.substring(i + 1, label.length() - 1);
                this.labels.add(substring);
            }
        }
    }

    public void classifyImage(BufferedImage image, int inputSize) {
//        ByteArrayOutputStream imageStream = new ByteArrayOutputStream();
//        try {
//            ImageIO.write(image, "jpg", imageStream);
//        } catch (IOException e) {
//            throw new RuntimeException("Bad image conversion to byteArray.");
//        }

//        Tensor<Float> imageTensor = normalizeImage(imageStream.toByteArray());
        Tensor<UInt8> imageTensor = normalizeImage_UInt8(image, inputSize);

        Detection detection = executeGraph(imageTensor);

        processDetections(detection, inputSize);
    }

//    /**
//     * Pre-process input. It resize the image and normalize its pixels
//     * @param imageBytes Input image
//     * @return Tensor<Float> with shape [1][416][416][3]
//     */
//    public static Tensor<Float> normalizeImage(final byte[] imageBytes) {
//        try (Graph graph = new Graph()) {
//            GraphBuilder graphBuilder = new GraphBuilder(graph);
//
//            final Output<Float> output =
//                    graphBuilder.div( // Divide each pixels with the MEAN
//                            graphBuilder.resizeBilinear( // Resize using bilinear interpolation
//                                    graphBuilder.expandDims( // Increase the output tensors dimension
//                                            graphBuilder.cast( // Cast the output to Float
//                                                    graphBuilder.decodeJpeg(
//                                                            graphBuilder.constant("input", imageBytes), 3),
//                                                    Float.class),
//                                            graphBuilder.constant("make_batch", 0)),
//                                    graphBuilder.constant("size", new int[]{SIZE, SIZE})),
//                            graphBuilder.constant("scale", MEAN));
//
//            try (Session session = new Session(graph)) {
//                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
//            }
//        }
//    }

    public Tensor<UInt8> normalizeImage_UInt8(BufferedImage image, int inputSize) {
        int[] imageInts = new int[inputSize * inputSize];
        byte[] byteValues = new byte[inputSize * inputSize * 3];

        image.getRGB(0,0, image.getWidth(), image.getHeight(), imageInts, 0, image.getWidth());
//
        for (int i = 0; i < imageInts.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (imageInts[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((imageInts[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((imageInts[i] >> 16) & 0xFF);
        }

//        inputName, byteValues, 1, inputSize, inputSize, 3
        long[] dims = new long[] {1, inputSize, inputSize, 3};
        return Tensor.create(UInt8.class, dims, ByteBuffer.wrap(byteValues));
    }

    /**
     * Executes graph on the given preprocessed image
     * @param image preprocessed image
     * @return output tensor returned by tensorFlow
     */
    private Detection executeGraph(final Tensor<?> image) {
        try (Graph graph = loadGraph()) {

            try(CustomClassifier classifier = new CustomClassifier(graph)) {
                classifier.feed(INPUT_NAME, image);
                classifier.run();

                float[] num_detections = classifier.get_num_detections();
                float[] detection_boxes = classifier.get_detection_boxes();
                float[] detection_scores = classifier.get_detection_scores();
                float[] detection_classes = classifier.get_detection_classes();

                Detection detection = new Detection(num_detections, detection_boxes, detection_scores, detection_classes);

                System.out.println(num_detections);

                String s = "";

                return detection;
            }
        }
    }

    private Graph loadGraph() {
        Graph graph = new Graph();
        graph.importGraphDef(graphBytes);
        return graph;
    }

    private void processDetections(Detection detection, int inputSize) {
        // Find the best detections.
        final PriorityQueue<Recognition> priorityQueue =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition recognition1, final Recognition recognition2) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(recognition2.getConfidence(), recognition1.getConfidence());
                            }
                        });

//        outputLocations); //detection_boxes
//        outputScores); //detection_scores
//        outputClasses); //detection_classes
//        outputNumDetections); //num_detections

        float[] detection_boxes = detection.getDetection_boxes();
        float[] detection_scores = detection.getDetection_scores();
        float[] detection_classes = detection.getDetection_classes();

        // Scale them back to the input size.
        for (int i = 0; i < detection_scores.length; ++i) {
            final RectFloats rectDetection =
                    new RectFloats(
                            detection_boxes[4 * i + 1] * inputSize,
                            detection_boxes[4 * i] * inputSize,
                            detection_boxes[4 * i + 3] * inputSize,
                            detection_boxes[4 * i + 2] * inputSize);
            priorityQueue.add(
                    new Recognition("" + i, labels.get(((int) detection_classes[i]) - 1), detection_scores[i], rectDetection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
            recognitions.add(priorityQueue.poll());
        }

        String s = "hoho";
    }

//    private String[] fetchNames() {
//
//    }
}
