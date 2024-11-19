import java.util.Random;
import java.lang.Math;

public class Datasets {
    
    int n=10; // the number of points to be generated
    

    //constructor :
    public Datasets(int input_int){
        n = input_int;
    }


    
    static void shuffle(Points2D points){
        
    }
    /**
     * shuffles Points2D 
     * @param points
     * @return shuffled points set
     */
    

    
    
    public Points2D DatasetA (){
        Datasets object = new Datasets(n-4);
        Points2D points = object.DatasetB();
        //adding the 4 corners
        points.add(new Point2D(-1, 1));
        points.add(new Point2D(-1, -1));
        points.add(new Point2D(1,-1));
        points.add(new Point2D(1,1)); 
        
        //rotating the points
        Random rand = new Random();
        double rand_angle=  rand.nextDouble()*2*Math.PI; 
        for (Point2D k : points){
            k.rotate(rand_angle);
        }
        //shuffling the set of points
        shuffle(points);
        return points;
    }


    
    public Points2D DatasetB (){
        Points2D points = new Points2D(n);
        for (int i=0; i<n;i++){
            Random rand = new Random();
            points.add(new Point2D (rand.nextDouble(-1,1),rand.nextDouble(-1,1)));            
        }
        return points;
    }


    
    public Points2D DatasetC (){ 
        Points2D points = new Points2D(n);
        int i=0;
        while (i<n) {
            Random rand = new Random();
            double x = rand.nextDouble(-1,1);
            double y = rand.nextDouble(-1,1);
            if (x*x+y*y<1) {
                points.add(new Point2D(x,y));
                i++;
            } 
        }
        return points;
    }


    
    public Points2D DatasetD (){ 
        Points2D points = new Points2D(n);
        for (int i=0; i<n-4;i++){
            Random rand = new Random();
            double teta=rand.nextDouble()*2*Math.PI;
            points.add(new Point2D(Math.cos(teta),Math.sin(teta)));
        }
        return points;
    }




   
    //we will be displaying the generated points in the datasets
    //for that we will create a class display :
    
    



    public static void main (String[] args){
        Datasets points = new Datasets(10);
        Points2D pointsA = points.DatasetC();
        pointsA.Cout();
        

    }
}
