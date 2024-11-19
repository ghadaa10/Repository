public class Segment {
    double x1, y1, x2, y2;
    boolean visible;

    public Segment(Point2D p1, Point2D p2) {
        x1 = p1.x;
        y1 = p1.y;
        x2 = p2.x;
        y2 = p2.y;
        visible = false;
    }
}
