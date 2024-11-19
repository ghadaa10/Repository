import java.util.*;

public class Points2D implements Iterable<Point2D>{
    ArrayList<Point2D> points; 

    //constructor
    public Points2D(int len){
        points = new ArrayList<Point2D>(len);
    }

    

    @Override
    public Iterator<Point2D> iterator(){
        return points.iterator();
    }
    // Method to add an element in the points
    public void add(Point2D p){
        points.add(p);
    }   
    // Method to check if the list is empty
    public boolean isEmpty(){
        if (points.size()==0) return true;
        return false;
    }
    // Method to measure the size of points
    public int length(){
        return points.size();
    }
    // Method to change the element corresponding to the index with an other point
    public void set(int index, Point2D point){
        points.set(index, point) ;
    }
    // Method to get the element corresponding to the index  
    public Point2D get(int index){
        return points.get(index);
    }
    // Method to sort the Arraylist
    public void sortArray2D(){
        points.sort((a,b)->Double.compare(a.x, b.x)); //the use of the lambda function is much more intuitive
    }
    // Method to remove a point from points
    public void remove(Point2D k){
        points.remove(k);
    }
    //Method to concatenate a Points2D list in the end
    public void addAll(Points2D k){
        points.addAll(k.points);
    }
    // Method to insert a point at index in a list p
    public void insertPoint(int index, Point2D p){
        points.add(index, p);
    }
    // Method to print the elements of points
    public void Cout(){
        if (points.isEmpty()) System.out.println("empty");
        for (Point2D point : points){
            System.out.println(point.toString());
        }
    }
    // Method to get the point with the maximum y coordinate
    public Point2D maxY(){
        Point2D max = points.get(0);
        for (Point2D p:points){
            if (max.y<p.y) max = p;
        }
        return max;
    }
    // Method to get the point with minimum y coordinate
    public Point2D minY(){
        Point2D min = points.get(0);
        for (Point2D p:points){
            if (min.y>p.y) min = p;
        }
        return min;
    }
    // Method to get the point with maximum X coordinate
    public Point2D maxX(){
        Point2D max = points.get(0);
        for (Point2D p:points){
            if (max.x<p.x) max = p;
        }
        return max;
    }
    // Method to get the point with minimum x coordinate
    public Point2D minX(){
        Point2D min = points.get(0);
        for (Point2D p:points){
            if (min.x>p.x) min = p;
        }
        return min;
    }
    // Method to make a copy of the list
    public Points2D copy(){
        Points2D copied = new Points2D(points.size());
        for (Point2D k :points){
            copied.add(new Point2D(k.x, k.y));
        }
        return copied;
    }
    // Method to remove the last element of the list
    public void pop(){
        if (!points.isEmpty()) {
            points.remove(points.size()-1);
        }
    }
    // Method to get a reversed list
    public void reverse(){
        ArrayList<Point2D> L= new ArrayList<Point2D>(points.size());
        for (int i=points.size()-1;i>=0;i--){
            L.add(points.get(i));
        }
        points=L;
    }
    // Method to get two points randomly, one in the left of xm and the other in the right of xm
    public Points2D pickRand_left_right(double xm){
        Random rand = new Random();
        boolean left = true;
        boolean right = true;
        Point2D p_left;
        Point2D p_right;
        Points2D result = new Points2D(2);
        while (left && right){
            Point2D first_pick = points.get(rand.nextInt(points.size()));
            if(first_pick.x<xm){
                left = false;
                p_left = first_pick;
                points.remove(p_left);
                result.add(p_left);
                while (right){
                    p_right = points.get(rand.nextInt(points.size()));
                    if (p_right.x>xm) {
                        right=false;
                        points.remove(p_right);
                        result.add(p_right);
                    }
                }
                
                
            }
            if(first_pick.x>xm){
                right = false;
                p_right = first_pick;
                points.remove(p_right);
                result.add(p_right);
                while (left){
                    p_left = points.get(rand.nextInt(points.size()));
                    if (p_left.x<xm) {
                        left=false;
                        points.remove(p_left);
                        result.add(p_left);
                    }
                }
            }
        }
        return result;
    }

    public static void main (String[] args){
        Points2D p = new Points2D(5);

        p.Cout();

    }




    
}
 