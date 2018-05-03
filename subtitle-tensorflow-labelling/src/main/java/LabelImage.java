import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
public class LabelImage {
    private static void printUsage(PrintStream s) {
        final String url =
                "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
        s.println(
                "Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)");
        s.println("to label JPEG images.");
        s.println("TensorFlow version: " + TensorFlow.version());
        s.println();
        s.println("Usage: label_image <model dir> <image file>");
        s.println();
        s.println("Where:");
        s.println("<model dir> is a directory containing the unzipped contents of the inception model");
        s.println("            (from " + url + ")");
        s.println("<image file> is the path to a JPEG image file");
    }

    public static void main(String[] args) {
//        if (args.length != 2) {
//            printUsage(System.err);
//            System.exit(1);
//        }
//        String modelDir = args[0];
//        String imageFile = args[1];
        String modelDir = "/home/jacob/andet/training/docker-training-shared/subs/trained-files";
//        String imageFile = "/home/jacob/andet/training/docker-training-shared/subs/sub_photos/sub/" + "00008.jpg";
        String imageFile = "/home/jacob/andet/training/docker-training-shared/subs/" + "3241-0_original.jpg";


        byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "output_graph.pb"));
        List<String> labels =
                readAllLinesOrExit(Paths.get(modelDir, "output_labels.txt"));
        byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));

        try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int bestLabelIdx = maxIndex(labelProbabilities);
            System.out.println(
                    String.format("BEST MATCH: %s (%.2f%% likely)",
                            labels.get(bestLabelIdx),
                            labelProbabilities[bestLabelIdx] * 100f));
        }
    }

    private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
        try (Graph graph = new Graph()) {
            GraphBuilder builder = new GraphBuilder(graph);
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.
            final int Height = 299;
            final int Width = 299;
            final float mean = 0f;
            final float scale = 255f;

            // Since the graph is being constructed once per execution here, we can use a constant for the
            // input image. If the graph were to be re-used for multiple input images, a placeholder would
            // have been more appropriate.
            final Output<String> input = builder.constant("input", imageBytes);

            final Output<Float> resizedImage = builder.resizeBilinear(
                    builder.expandDims(
                            builder.cast(builder.decodeJpeg(input, 3), Float.class),
                            builder.constant("make_batch", 0)),
                    builder.constant("size", new int[] {Height, Width}));

            final Output<Float> output =
                    builder.div(
                            builder.sub(resizedImage, builder.constant("mean", mean)),
                            builder.constant("scale", scale));
            try (Session session = new Session(graph)) {
                // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class); // expect casts the Tensor<?> to Tensor<Float>.
            }
        }
    }

    private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
                 Tensor<Float> result =
                         s.runner()
                                 .feed("Mul", image)
                                 .fetch("final_result")
                                 .run()
                                 .get(0)
                                 .expect(Float.class)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }


}
