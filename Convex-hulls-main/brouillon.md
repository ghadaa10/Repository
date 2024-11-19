Now that's a discovery.
There's already a class in java that does the work and implements Graham's algorithm aka the sweeping method.
It's GrahamScan.getConvexhull : 
List<java.awt.Point> convexHull = GrahamScan.getConvexHull(xs, ys); 

also we can change the datastructure for points : simply using java.awt.Point :
List<java.awt.Point> points = Arrays.asList(
        new java.awt.Point(x1,y1),
        ...
);

