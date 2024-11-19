public class Point2D {
    double x;
    double y;

    //constructor
    Point2D(double x,double y){
        this.x = x;
        this.y = y;
    }

    public String toString(){
        return String.valueOf(this.x)+" "+String.valueOf(this.y);
    }

    public boolean isEqual(Point2D p){
        return ((this.x==p.x)&&(this.y == p.y));
    }

    public void rotate(double angle) {
        double newX = x * Math.cos(angle) - y * Math.sin(angle);
        double newY = x * Math.sin(angle) + y * Math.cos(angle);
        x = newX;
        y = newY;
    }

    
}