import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.util.ArrayList;

/* this class is acting as a canvis to display the generated datasets
 * it is called in display_datasets and takes as arguments the width 
 * and height of the panel as well as the array containing the coordinates 
 * x and y of the generated points 
 */

public class datasets_canvas extends JComponent {
    private static  int width;
    private static int height;
    private double marg_h;
    private double marg_v;
    private Points2D coordinates;
    public ArrayList<Segment> segments = new ArrayList<>(
    );

    //constructor:
    public datasets_canvas(int w,int h,Points2D points,Points2D convex){
        width = w;
        height = h;
        marg_h = w*0.01;
        marg_v = h*0.05;
        coordinates = points.copy();
        for (int i=1; i<convex.length();i++){
            segments.add(new Segment(convex.get(i-1), convex.get(i)));
        }
    }

    public static double transformX(double x, double scale, double marg_h){
        return marg_h + (x+Math.sqrt(2))*scale*(height-2*marg_h);
    }
    public static double transformY(double y, double scale, double marg_v){
        return marg_v + (-y+Math.sqrt(2))*scale*(height-2*marg_v);
    }

    
    public void paintComponent(Graphics g){
        Graphics2D g2 = (Graphics2D) g;
        //Graphics2D has interesting features like anti-aliasing
        //it will allow us to have smoother circles!
        g2.setRenderingHint( RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON );
        //we draw our our sqaure:
        g2.draw(new Line2D.Double(marg_h,marg_v,marg_h, height-marg_v));
        g2.draw(new Line2D.Double(marg_h,height-marg_v,width-marg_h, height-marg_v));
        g2.draw(new Line2D.Double(marg_h,marg_v,width-marg_h,marg_v));
        g2.draw(new Line2D.Double(width-marg_h,marg_v,width-marg_h,height-marg_v));
        
        
        //Ellipse2D.Double r = new Ellipse2D.Double(50,50,400,400);


        /*we plot the points in the sqaure we drew:
        since x and y are in [0,1] we will multiply them by our scale*/



        double scale = 1/(2*Math.sqrt(2));
        

        double r = 3; //radius of the points


    

        for (Point2D p : coordinates){
            g2.fill(new Ellipse2D.Double(transformX(p.x, scale, marg_h)-r,transformY(p.y, scale, marg_v)-r,2*r,2*r)); //we print the dots as little circles of radius 2 using the ellipse shape
        }

        for (Segment s : segments) {
            if (s.visible){
                g2.draw(new Line2D.Double(transformX(s.x1, scale, marg_h), transformY(s.y1, scale, marg_v), transformX(s.x2, scale, marg_h), transformY(s.y2, scale, marg_v)));
            }
            
        }




        g2.setColor(new Color(255,255,255));
        
    }

    public static void main (String[] args){

    }
}
