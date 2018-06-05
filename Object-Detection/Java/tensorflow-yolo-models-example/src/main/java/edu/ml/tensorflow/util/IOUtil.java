package edu.ml.tensorflow.util;

import edu.ml.tensorflow.ObjectDetector;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Util class to read image, graphDef and label files.
 */
public final class IOUtil {
    private final static Logger LOGGER = LoggerFactory.getLogger(IOUtil.class);
    private IOUtil() {}

    public static byte[] readAllBytesOrExitLocal(final String fileName) {
        try {
            return IOUtils.toByteArray(ObjectDetector.class.getResourceAsStream(fileName));
        } catch (IOException | NullPointerException ex) {
            LOGGER.error("Failed to read [{}]!", fileName);
            throw new ServiceException("Failed to read [" + fileName + "]!", ex);
        }
    }

    public static byte[] readAllBytesOrExit(final String fileName) {
        try {
            Path filePath = Paths.get(fileName);
            return IOUtils.toByteArray(Files.newInputStream(filePath));
        } catch (IOException | NullPointerException ex) {
            LOGGER.error("Failed to read [{}]!", fileName);
            throw new ServiceException("Failed to read [" + fileName + "]!", ex);
        }
    }

    public static List<String> readAllLinesOrExit(final String filename) {
        try {
            File file = new File(ObjectDetector.class.getResource(filename).toURI());
            return Files.readAllLines(file.toPath(), Charset.forName("UTF-8"));
        } catch (IOException | URISyntaxException ex) {
            LOGGER.error("Failed to read [{}]!", filename, ex.getMessage());
            throw new ServiceException("Failed to read [" + filename + "]!", ex);
        }
    }

    public static void createDirIfNotExists(final File directory) {
        if (!directory.exists()) {
            directory.mkdir();
        }
    }

    public static String getFileName(final String path) {
        return path.substring(path.lastIndexOf("/") + 1, path.length());
    }
}
