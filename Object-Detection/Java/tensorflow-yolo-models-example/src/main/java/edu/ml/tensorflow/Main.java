package edu.ml.tensorflow;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;

public class Main {

    public static void main(String[] args) {

//        Path imageDir = Paths.get("/image/");


        ObjectDetector objectDetector = new ObjectDetector();
        String imageDirString = objectDetector.getClass().getClassLoader().getResource("image").getPath();

        String decodedPath = "";
        try {
            decodedPath = URLDecoder.decode(imageDirString, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        Path imageDir = Paths.get(decodedPath);

        boolean directory = Files.isDirectory(imageDir);

        Collection<File> images = FileUtils.listFiles(imageDir.toFile(), new String[]{"jpg"}, false);

        for (File image : images) {
            objectDetector.detect(image.getAbsolutePath());
        }
    }
}
