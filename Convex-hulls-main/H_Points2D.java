import java.util.HashMap;
import java.util.Map.Entry;

public class H_Points2D {
    HashMap<String, Point2D> pointTable;

    public H_Points2D() {
        pointTable = new HashMap<>();
    }

    // Method to add a point to the table
    public void add(int key, Point2D point) {
        pointTable.put(String.valueOf(key), point);
    }

    // Method to access a point from the table
    public Point2D get(int key) {
        return pointTable.get(String.valueOf(key));
    }

    // Method to remove a point from the table
    public void remove(int key) {
        pointTable.remove(String.valueOf(key));
    }

    // Method to pop random point from the 'non empty' table and return it
    

    public void Cout(){
        for (Entry<String, Point2D> entry: pointTable.entrySet()){
            System.out.println(String.valueOf(entry.getKey())+" : " + entry.getValue().toString());
        }
    }

    

}
