import java.util.*;
import java.util.Random;

public class Main_Algo{
    //fields
    public Points2D points;
    private Points2D convexhull ;


    //constructor
    public Main_Algo(Points2D intialPoints){
        //we initialize by createing a copy of our set of points 
        points = intialPoints.copy();
        convexhull= new Points2D(intialPoints.length());

    }
        

   
    /**
     * @param to_be_split : list of points to be split into 2 sets left and right by the pivot
     * @param pivot : the pivot
     * @return list left of points where x < pivot, list right of points where x > pivot and list equal of points where x = pivot
     */
    public static ArrayList<Points2D> splitPivot (Points2D to_be_split, double pivot){
        int length = to_be_split.length();
        Points2D less = new Points2D(length);
        Points2D great = new Points2D(length);
        Points2D equal = new Points2D(length);
 
        for (Point2D k : to_be_split){
            if (k.x<pivot) {
                less.add(k);
   
            }
            if (k.x>pivot) {
                great.add(k);

            }
            if (k.x==pivot) {
                equal.add(k);

            }
        }
        ArrayList<Points2D> result = new ArrayList<>(3);
        result.add(less);
        result.add(equal);
        result.add(great);
        return result;

    }

    /**
     * @param points
     * @return the median of the x coordinates 
     */
    public static double Median(Points2D points){
        int len = points.length(); 
        
        if(len%2==1) return searchRank(points, (len)/2); //to be defined later either picked randomly or selected cleverly
        else return 0.5*(searchRank(points, len/2) + searchRank(points, len/2-1));
    }


    

    /**
     * @param points
     * @param k : the rank of the x coordinate
     * @return the kth smallest x coordinate in points 
     */
    public static double searchRank(Points2D point, int k){
        int length = point.length();
        if (length==1) return point.get(0).x; //the base condition
 

        //we choose a pivot randomly from our list :
        Random rand = new Random();
        int pivot_index = rand.nextInt(length);

        //we split our list using the splitPivot method into points where x is less, equal and greater than the pivot respectively
        
        ArrayList <Points2D> pivot_split_list = splitPivot(point, point.get(pivot_index).x);
        Points2D less = pivot_split_list.get(0);
        Points2D equal = pivot_split_list.get(1);
        Points2D great = pivot_split_list.get(2);

        if (k<less.length()) {
            return searchRank(less, k);
        }
        else if (k<less.length() + equal.length()) {return equal.get(0).x;}
        else {return searchRank(great, k-less.length()-equal.length());}

    }




    /**
     * @param subset : it can be either the left set or the right set of points depending on the position of p0 to the median
     * @param p0 : The point of reference
     * @param xm : the median of the x coordinates
     * @return a point A such that the intersection of (p0,A) with x=xm has the highest coordinate
     */
    public static Point2D arch_point(Points2D subset,Point2D p0, double xm, boolean isMax){
        Point2D A = subset.get(0);
        double ext_y=0;
        for (Point2D k : subset){
            double pente=(k.y-p0.y)/(k.x-p0.x);
            double c=p0.y-pente*p0.x;
            // we search the point of the intersection where y is maximal
            boolean isUpdated = false;
            if (isMax){
                if (ext_y < pente*xm + c) isUpdated = true;
            } else {
                if (ext_y > pente*xm + c) isUpdated = true;
            }
            if (isUpdated){
                //System.out.println("ext before"+String.valueOf(ext_y));
                //System.out.println("update? "+String.valueOf(isUpdated));
                ext_y=pente * xm + c;
                A=k;
                //System.out.println("ext after"+String.valueOf(ext_y));
                //System.out.println("points"+A.toString();
            } 
        }
        return(A);
    }



    
    /**
     * @param points : the initial set of points 
     * @return the pair of points forming the highest line that intersects the vertical line of equation x = xm as low as possible 
     */
    public static Points2D bridgeconstruction(Points2D points, boolean isMax){
        
        //we get the median
        double xm=Median(points);
        //then we split the points set into 3 parts
        ArrayList <Points2D> pivot_split_list = splitPivot(points, xm);
        Points2D less = pivot_split_list.get(0);
        Points2D equal = pivot_split_list.get(1);
        Points2D great = pivot_split_list.get(2);
        
 
        
        //first we will determine the bridge following the suggested algorirhm, then we will discuss the case where length(equal) is not 0 


        // 1. we pick a left point and a right point randomly
        Random rand = new Random();
        int index_left = rand.nextInt(less.length());
        int index_right = rand.nextInt(less.length());
        Point2D p_left=less.get(index_left);
        Point2D p_right=great.get(index_right);
        less.remove(p_left);
        great.remove(p_right);
        //System.out.println("l = "+p_left.toString()+"// r= "+p_right.toString());

        //2. we initialize Pk that we naturally split into lesser set and greater set..
        Points2D new_less = new Points2D(less.length());
        Points2D new_great = new Points2D(great.length());
        new_less.add(p_left);
        new_great.add(p_right);

        // then pick a random point either from less or great with equal probability
        while (less.length()+great.length()>0){
            Point2D pk ; //the point to be added
            double proba = rand.nextDouble();
            if(less.isEmpty()){
                pk = great.get(rand.nextInt(great.length()));
                great.remove(pk);
            }else if (great.isEmpty()){
                pk = less.get(rand.nextInt(less.length()));
                less.remove(pk);
            }else if (proba<0.5) {
                pk = less.get(rand.nextInt(less.length()));
                less.remove(pk);
            }else{
                pk = great.get(rand.nextInt(great.length()));
                great.remove(pk);
            }
            // then we add it to either new_less or new_great
            if (pk.x<xm){
                new_less.add(pk);
            }else{
                new_great.add(pk);
            }


            
            //System.out.println("new less ");
            //new_less.Cout();
            //System.out.println("new great ");
            //new_great.Cout();
            //System.out.println("picked point = "+pk.toString());
        //then update p_left and p_right accordingly

            String direction = "clockwise";
            if (isMax) direction = "counterclockwise";
            boolean condition = sweeping.triangleDirection(p_left,p_right,pk)==direction;
            //System.out.println(condition);
            if (condition) {
                if (pk.x<xm){
                    p_left=pk;
                    p_right=arch_point(new_great,pk,xm, isMax);
                    less.remove(p_right);
                    //System.out.println("p_left = "+p_left.toString()+"// arch= "+p_right.toString());

                }
                else {
                    p_right=pk;
                    p_left=arch_point(new_less,pk,xm, isMax);
                    less.remove(p_left);
                    //System.out.println("p_right = "+p_right.toString()+"// arch= "+p_left.toString());
                }
            }
        }
        //3. now we'll discuss the cases where equal is NOT empty
        if (equal.length()>0){
            //we check if the point with the highest y in equal is above or under the bridge we found
            if (isMax){
                Point2D max = equal.maxY();
                boolean condition = sweeping.triangleDirection(p_left, p_right, max)=="counterclockwise";
                if (condition){
                    p_left = max;
                    p_right= max;
                }
            }else{
                Point2D min = equal.maxY();
                if (sweeping.triangleDirection(p_left, p_right, min)=="clockwise"){
                    p_left = min;
                    p_right= min;
                }
            }
             
        }

        // we finally construct the bridge and return it
        Points2D bridge = new Points2D(2);
        bridge.add(p_left);
        bridge.add(p_right);
        return bridge;
    }




    public static Points2D halfHull(Points2D points, boolean isMax){
        //base case : 
        if (points.length()<=1) {
            return(points);
        }

        //System.out.println("points = ");
        //points.Cout();

        Points2D bridge = bridgeconstruction(points, isMax);
        Point2D l=bridge.get(0);
        Point2D r=bridge.get(1); 
        //System.out.println("l = "+l.toString()+"//  r = "+r.toString());

        //split the points outside of the region under the bridge
        Points2D left_points = splitPivot(points, l.x).get(0); //points left of l
        Points2D right_points = splitPivot(points, r.x).get(2); //points right of r
        left_points.add(l);
        right_points.add(r);
        //System.out.println("left side after split");
        //left_points.Cout();
        //System.out.println("right side after split");
        //right_points.Cout();
         
        Points2D lefthull= halfHull(left_points,isMax);
        Points2D righthull=halfHull(right_points,isMax);
        
        if(lefthull.get(lefthull.length()-1)==righthull.get(0)){
            lefthull.pop();
        }
        
        lefthull.addAll(righthull);
        //System.out.println("the hull :"); 
        return(lefthull);       
    }


    public Points2D convexHull_construction(Points2D points){
        Points2D upperHull = halfHull(points, true);
        Points2D lowerHull = halfHull(points, false);
        convexhull = upperHull;
        for (int i =lowerHull.length()-2;i>0;i--){
            convexhull.add(lowerHull.get(i));
        }
        convexhull.add(upperHull.get(0));
        return convexhull;
    }


   











    public static void main (String[] args){
        Datasets points = new Datasets(10);
        Points2D pointsA = points.DatasetB();
        pointsA.Cout();
        bridgeconstruction(pointsA, true).Cout();
        

        

        

    }
}