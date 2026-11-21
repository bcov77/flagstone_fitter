import java.io.*;
import java.util.Locale;
import org.locationtech.jts.geom.*;
import org.locationtech.jts.io.WKTReader;
import org.locationtech.jts.algorithm.construct.LargestEmptyCircle;

/** Candidate oracle only. Python rechecks all radii; this emits no certificate. */
public class CornerGapProbe {
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.ROOT);
    try (BufferedReader reader = new BufferedReader(new FileReader(args[0]))) {
      WKTReader wkt = new WKTReader();
      String line;
      while ((line = reader.readLine()) != null) {
        String[] fields = line.split("\t");
        long started = System.nanoTime();
        LargestEmptyCircle lec = new LargestEmptyCircle(wkt.read(fields[2]), wkt.read(fields[3]), Double.parseDouble(fields[1]));
        Point center = lec.getCenter();
        System.out.printf("%s\t%.17g\t%.17g\t%.17g\t%.9f%n", fields[0],center.getX(),center.getY(),
            lec.getRadiusLine().getLength(),(System.nanoTime()-started)/1e9);
      }
    }
  }
}
