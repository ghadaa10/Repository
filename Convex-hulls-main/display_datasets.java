import java.util.ArrayList;

import javax.swing.*;



public class display_datasets {
    datasets_canvas display;
    Points2D points;
    Points2D convex;
    int width;
    int height;

    //constructor:

    public display_datasets(Points2D p,Points2D con ,int w, int h){
        this.points = p;
        this.convex = con;
        width = w;
        height = h;
        display = new datasets_canvas(width, height, points,convex);
    }

    public void displayPoints (){

        JFrame frame = new JFrame();
        frame.setSize(width,height);
        frame.setTitle("Convexhull");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true); 

        frame.add(display);

    }

    public void displayConvex(){
        JFrame frame = new JFrame();

        frame.setSize(width,height);
        frame.setTitle("Convexhull");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true); 

        frame.add(display);

        ArrayList <Segment> segments = display.segments;

        for (int i = 0; i < segments.size(); i++) {
            final int index = i;
            try {
                Thread.sleep(250);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    display.segments.get(index).visible = true;
                    //System.out.println("is it visible? " + String.valueOf(display.segments.get(index).visible));
                    display.repaint();
                }
            });
        }
    }



}
