package org.tensorflow.demo.custom;

import org.apache.commons.io.FileUtils;
import org.tensorflow.*;
import org.tensorflow.demo.GraphBuilder;
import org.tensorflow.types.UInt8;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;

public class CustomObjectDetector {

    final static int SIZE = 200;
    final static float MEAN = 255f;
    final static String INPUT_NAME = "image_tensor";

    private byte[] graphBytes;

    public void addGraph(File graphFile) throws IOException {
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

    public void classifyImage(BufferedImage image, int inputSize) {
//        ByteArrayOutputStream imageStream = new ByteArrayOutputStream();
//        try {
//            ImageIO.write(image, "jpg", imageStream);
//        } catch (IOException e) {
//            throw new RuntimeException("Bad image conversion to byteArray.");
//        }

//        Tensor<Float> imageTensor = normalizeImage(imageStream.toByteArray());
        Tensor<UInt8> imageTensor = normalizeImage_UInt8(image, inputSize);

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
    private float[] executeGraph(final Tensor<?> image) {
        try (Graph graph = loadGraph()) {

            try(CustomClassifier classifier = new CustomClassifier(graph)) {
                classifier.feed(INPUT_NAME, image);
                classifier.run();

                float[] num_detections = classifier.get_num_detections();
                float[] detection_boxes = classifier.get_detection_boxes();

                System.out.println(num_detections);

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
