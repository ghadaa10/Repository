public class Main {
public static void display_sweeping_datasetA(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsA = points.DatasetA();
  sweeping obj = new sweeping(pointsA);
  obj.sweepingAlgo();
  Points2D convex = obj.get();
  display_datasets display = new display_datasets(pointsA, convex, w, h);
  //display.displayConvex();
}
public static void display_sweeping_datasetB(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsB = points.DatasetB();
  sweeping obj = new sweeping(pointsB);
  obj.sweepingAlgo();
  Points2D convex = obj.get();
  display_datasets display = new display_datasets(pointsB, convex, w, h);
  //display.displayConvex();
}
public static void display_sweeping_datasetC(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsC = points.DatasetC();
  sweeping obj = new sweeping(pointsC);
  obj.sweepingAlgo();
  Points2D convex = obj.get();
  display_datasets display = new display_datasets(pointsC, convex, w, h);
  //display.displayConvex();
}
public static void display_sweeping_datasetD(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsD = points.DatasetD();
  sweeping obj = new sweeping(pointsD);
  obj.sweepingAlgo();
  Points2D convex = obj.get();
  display_datasets display = new display_datasets(pointsD, convex, w, h);
  //display.displayConvex();
}

public static void display_MainAlgo_datasetA(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsA = points.DatasetA();
  Main_Algo obj = new Main_Algo(pointsA);
  Points2D convex = obj.convexHull_construction(pointsA);
  display_datasets display = new display_datasets(pointsA, convex, w, h);
  //display.displayConvex();
}
public static void display_MainAlgo_datasetB(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsB = points.DatasetB();
  Main_Algo obj = new Main_Algo(pointsB);
  Points2D convex = obj.convexHull_construction(pointsB);
  display_datasets display = new display_datasets(pointsB, convex, w, h);
  //display.displayConvex();
}
public static void display_MainAlgo_datasetC(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsC = points.DatasetC();
  Main_Algo obj = new Main_Algo(pointsC);
  Points2D convex = obj.convexHull_construction(pointsC);
  display_datasets display = new display_datasets(pointsC, convex, w, h);
  //display.displayConvex();
}
public static void display_MainAlgo_datasetD(int n){
  int w = 1000; //we choose the width and height of our panel
  int h = 1000;
  Datasets points = new Datasets(n);
  Points2D pointsD = points.DatasetD();
  Main_Algo obj = new Main_Algo(pointsD);
  Points2D convex = obj.convexHull_construction(pointsD);
  display_datasets display = new display_datasets(pointsD, convex, w, h);
  //display.displayConvex();
}

    


        
    public static void main(String[] args){
   
        int n = 50; //number of generated points
      
        
        

        long start = System.nanoTime();
        display_sweeping_datasetA(n);
        long end = System.nanoTime();
        long time = end - start;
        String result= "the time of sweeping Algo on DatasetA :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_sweeping_datasetB(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of sweeping Algo on DatasetB :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_sweeping_datasetC(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of sweeping Algo on DatasetC :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_sweeping_datasetD(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of sweeping Algo on DatasetD :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_MainAlgo_datasetA(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of Main Algo on DatasetA :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_MainAlgo_datasetB(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of Main Algo on DatasetB :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_MainAlgo_datasetC(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of Main Algo on DatasetC :"+Long.toString(time)+"\n";

        start = System.nanoTime();
        display_MainAlgo_datasetD(n);
        end = System.nanoTime();
        time = end - start;
        result+= "the time of Main Algo on DatasetD :"+Long.toString(time)+"\n";
      
        System.out.print(result);
      
      // uncomment the display option in the code above to visualise the generation of the convex hull for both algorithms on each data set:
      // here's an example:
      int w = 1000; //we choose the width and height of our panel
      int h = 1000;
      Datasets points = new Datasets(n);
      Points2D pointsB = points.DatasetB();
      Main_Algo obj = new Main_Algo(pointsB);
      Points2D convex = obj.convexHull_construction(pointsB);
      display_datasets display = new display_datasets(pointsB, convex, w, h);
      display.displayConvex();

      
     




    


        


       


        

    
    }
}
