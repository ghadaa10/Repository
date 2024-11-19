import java.util.*;


public class sweeping {
    private Stack<Point2D> convexHull = new Stack<Point2D>();
    public Points2D points ;

    //constructor
    public sweeping(Points2D intialPoints){
        //we initialize by createing a copy of our set of points
        int len = intialPoints.length(); 
        points = new Points2D(len);
        for (int i=0;i<len;i++){
            points.add(intialPoints.get(i)); 
        }
        //we sort our copied array
        points.sortArray2D();
        if (points.length()>1) {
        convexHull.push(points.get(0)); 
        convexHull.push(points.get(1)); 
        }

    }

    
    
    /**determines whether a triangle ABC is (counter)clockwise
     * @param a coordinates of A
     * @param b coordinates of B
     * @param c coordinates of C
     * @return "clockwise" or "counterclockwise" depending on whether ABC is clockwise or counterclockwise
     */
    public static String triangleDirection(Point2D a, Point2D b, Point2D c){
        double xAB = b.x-a.x;
        double yAB = b.y-a.y;
        double xAC = c.x-a.x;
        double yAC = c.y-a.y;
        double z = xAB*yAC - xAC*yAB; //we calculate the cross product projected on the z axis
        if (z>0) { 
            return "counterclockwise";
        }
        else if (z<0){
            return "clockwise";
        }
        return "alligned";
    }


    public void sweepingAlgo(){
        // we begin adding points from the third point
        for (int i=2;i<points.length();i++){
            Point2D k=points.get(i);
            int lastIndex = convexHull.size()-1;
            while ((lastIndex>=1)&&(triangleDirection(convexHull.get(lastIndex-1), convexHull.get(lastIndex), k))=="counterclockwise"){
                convexHull.pop();
                lastIndex--;                    
            }
            convexHull.push(k);      
        }            
        //we initialize the points in convexHull for sweepingAlgo_lower
        int n=convexHull.size();
        convexHull.push(points.get(points.length()-2));
        for (int i=points.length()-3;i>=0;i--){
            Point2D k=points.get(i);
            int lastIndex = convexHull.size()-1;
            while ((lastIndex>=n)&&(triangleDirection(convexHull.get(lastIndex-1), convexHull.get(lastIndex), k))=="counterclockwise"){
                convexHull.pop();
                lastIndex--;                    
            }
            convexHull.push(k);      
        }
             
    
        }
    public Points2D get(){
        Points2D P=new Points2D(convexHull.size());
        int n=convexHull.size();
        for (int i=0;i<n;i++){
            P.add(convexHull.pop());
        }
        P.reverse();
        return(P);
    }
    public static void main(String[] args){
        Datasets points = new Datasets(10);
        Points2D pointsA = points.DatasetC();
        pointsA.sortArray2D();
        System.out.println("this is A");
        for (Point2D p : pointsA){
        System.out.println(p.toString());
    }
        sweeping obj = new sweeping(pointsA);
        Points2D convex = obj.get();
        System.out.print("length()");
        System.out.print(convex.length());
        obj.sweepingAlgo();
        System.out.println("this is convex");
        convex = obj.get();
        for (Point2D p : convex){
            System.out.println(p.toString());
        }
    }
}



    




    

    
    
    

