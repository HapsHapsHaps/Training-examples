/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

//import android.content.res.AssetManager;
//import android.graphics.Bitmap;
//import android.graphics.RectF;
//import android.os.Trace;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.*;
import org.tensorflow.demo.contrib.TensorFlowInferenceInterface;
import org.tensorflow.demo.custom.CustomObjectDetector;
import org.tensorflow.types.UInt8;

import javax.imageio.ImageIO;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TensorFlowObjectDetectionAPIModel implements Classifier {
    private final static Logger log = LoggerFactory.getLogger(TensorFlowObjectDetectionAPIModel.class);

    // Only return this many results.
    private static final int MAX_RESULTS = 100;
    private static final int NUM_DETECTIONS = 2;
    // Params used for image processing
    int SIZE = 416;
    float MEAN = 255f;

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param modelFile The filepath of the model GraphDef protocol buffer.
     * @param labelFile The filepath of label file for classes.
     */
    public static Classifier create(
            final File modelFile,
            final File labelFile,
            final int inputSize) throws IOException {
        final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();

        InputStream labelsInput = Files.newInputStream(labelFile.toPath());
        BufferedReader reader = null;
        reader = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = reader.readLine()) != null) {
            log.warn(line);
            d.labels.add(line);
        }
        reader.close();

        log.debug("Classifier Labels: {}", d.labels.toString());

        InputStream modeFileInputStream = Files.newInputStream(modelFile.toPath());

        d.inferenceInterface = new TensorFlowInferenceInterface(modeFileInputStream);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(d.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
        }
        d.inputSize = inputSize;
        // The outputScoresName node has a shape of [N, NumLocations], where N
        // is the batch size.
        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }
        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }
        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }

        // Pre-allocate buffers.
        d.outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        d.intValues = new int[d.inputSize * d.inputSize];
        d.byteValues = new byte[d.inputSize * d.inputSize * 3];
        d.outputScores = new float[MAX_RESULTS];
        d.outputLocations = new float[MAX_RESULTS * 4];
        d.outputClasses = new float[MAX_RESULTS];
        d.outputNumDetections = new float[NUM_DETECTIONS];
        return d;
    }

    private TensorFlowObjectDetectionAPIModel() {}

    @Override
//    public List<Recognition> recognizeImage(final Bitmap bitmap) {
    public List<Recognition> recognizeImage(final BufferedImage bitmap) {
        // Log this method so that it can be analyzed with systrace.
//        Trace.beginSection("recognizeImage");

//        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
//        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//
//        ByteArrayOutputStream imageStream = new ByteArrayOutputStream();
//        try {
//            ImageIO.write(bitmap, "jpg", imageStream);
//        } catch (IOException e) {
//            throw new RuntimeException("Bad image conversion to byteArray.");
//        }
//        byteValues = image;

        bitmap.getRGB(0,0, bitmap.getWidth(), bitmap.getHeight(), intValues, 0, bitmap.getWidth());
//
        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
//        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
//        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);
//        inferenceInterface.addFeed(inputName, imageTensor);
//        Trace.endSection();

        // Run the inference call.
//        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
//        Trace.endSection();

        // Copy the output Tensor back into the output array.
//        Trace.beginSection("fetch");
        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(outputNames[0], outputLocations); //detection_boxes
        inferenceInterface.fetch(outputNames[1], outputScores); //detection_scores
        inferenceInterface.fetch(outputNames[2], outputClasses); //detection_classes
        inferenceInterface.fetch(outputNames[3], outputNumDetections); //num_detections
//        Trace.endSection();

        // Find the best detections.
        final PriorityQueue<Recognition> priorityQueue =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            final RectFloats detection =
                    new RectFloats(
                            outputLocations[4 * i + 1] * inputSize,
                            outputLocations[4 * i] * inputSize,
                            outputLocations[4 * i + 3] * inputSize,
                            outputLocations[4 * i + 2] * inputSize);
            priorityQueue.add(
                    new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
            recognitions.add(priorityQueue.poll());
        }
//        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
