package org.tensorflow.demo.custom;

import org.tensorflow.*;
import org.tensorflow.demo.Recognition;
import org.tensorflow.demo.RectFloats;
import org.tensorflow.types.UInt8;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
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
    private static final String INPUT_NAME = "image_tensor";
    private static final int MAX_RESULTS = 10;

    private byte[] graphBytes;
    private List<String> labels;

    public CustomObjectDetector(File graphFile, File labelFile) throws IOException {
        addGraph(graphFile);
        addLabels(labelFile);
    }

    private void addGraph(File graphFile) throws IOException {

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

    public ArrayList<Recognition> classifyImage(BufferedImage image) {

        int width = image.getWidth();
        int height = image.getHeight();
//        width = 200;
//        height = 200;

        Tensor<UInt8> imageTensor = normalizeImage_UInt8(image, width, height);

        Detection detection = executeGraph(imageTensor);

        return processDetections(detection, width, height);
//        test(detection);
//        return null;
    }

    public Tensor<UInt8> normalizeImage_UInt8(BufferedImage image, int width, int height) {
        int[] imageInts = new int[width * height];
        byte[] byteValues = new byte[width * height * 3];

//        image.getRGB(0,0, image.getWidth(), image.getHeight(), imageInts, 0, image.getWidth());

//        for (int i = 0; i < imageInts.length; ++i) {
//            byteValues[i * 3 + 2] = (byte) (imageInts[i] & 0xFF);
//            byteValues[i * 3 + 1] = (byte) ((imageInts[i] >> 8) & 0xFF);
//            byteValues[i * 3 + 0] = (byte) ((imageInts[i] >> 16) & 0xFF);
//        }

        if (image.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new RuntimeException("Expected 3-byte BGR encoding in BufferedImage, found " + image.getType() + ". This code could be made more robust");
        }

        byte[] data = ((DataBufferByte) image.getData().getDataBuffer()).getData();
        bgr2rgb(data);

//        inputName, byteValues, 1, inputSize, inputSize, 3
//        long[] dims = new long[] {1, width, height, 3};
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[] {BATCH_SIZE, image.getHeight(), image.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
//        return Tensor.create(UInt8.class, dims, ByteBuffer.wrap(byteValues));
    }

    /**
     * Converts image pixels from the type BGR to RGB
     * @param data
     */
    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
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


//                float[] num_detections = classifier.get_num_detections();
//                float[] detection_boxes = classifier.get_detection_boxes();
//                float[] detection_scores = classifier.get_detection_scores();
//                float[] detection_classes = classifier.get_detection_classes();

//                Detection detection = new Detection(num_detections, detection_boxes, detection_scores, detection_classes);

//                System.out.println(num_detections);

                return classifier.detections();
            }
        }
    }

    private Graph loadGraph() {
        Graph graph = new Graph();
        graph.importGraphDef(graphBytes);
        return graph;
    }

    private ArrayList<Recognition> processDetections(Detection detection, int width, int height) {
        // Find the best detections.
        final PriorityQueue<Recognition> priorityQueue =
                new PriorityQueue<Recognition>(1, new RecognitionComparetor());

        float[] detection_boxes = detection.getDetection_boxes();
        float[] detection_scores = detection.getDetection_scores();
        float[] detection_classes = detection.getDetection_classes();

        // Scale them back to the input size.
        for (int i = 0; i < detection_scores.length; ++i) {
            final RectFloats rectDetection =
                    new RectFloats(
                            detection_boxes[4 * i + 1] * width,
                            detection_boxes[4 * i] * height,
                            detection_boxes[4 * i + 3] * width,
                            detection_boxes[4 * i + 2] * height);
            priorityQueue.add(
                    new Recognition("" + i, labels.get(((int) detection_classes[i]) - 1), detection_scores[i], rectDetection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
            recognitions.add(priorityQueue.poll());
        }

        return recognitions;
    }

    /**
     * Used to make sure the detections with highest confidence, is placed highest in queue.
     */
    class RecognitionComparetor implements Comparator<org.tensorflow.demo.Recognition> {
        @Override
        public int compare(final org.tensorflow.demo.Recognition recognition1, final org.tensorflow.demo.Recognition recognition2) {
            // Intentionally reversed to put high confidence at the head of the queue.
            return Float.compare(recognition2.getConfidence(), recognition1.getConfidence());
        }
    }
}
