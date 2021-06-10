import com.firefly.ai.Bp;
import com.firefly.ai.listener.FitListener;

public class Test1 {
    public static void main(String[] args){
        double[][] x={
                {0,0,0},
                {0,0,1},
                {0,1,0},
                {0,1,1},
                {1,0,0},
                {1,0,1},
                {1,1,0},
                {1,1,1},
        };
        double[][] y={
                {1,1},
                {0,0},
                {0,0},
                {0,0},
                {0,0},
                {0,0},
                {0,0},
                {1,1},
        };

        Bp bp=new Bp(0.1,3,10,2);
        bp.fit(x, y, 1, 1000000, new FitListener() {
            public void onLoss(double loss) {
                System.out.println(loss);
            }
        });

        for(int i=0;i<x.length;i++){
            print(y[i],bp.predict(x[i]));
        }
    }

    private static void print(double[] label,double[] val){
        for(int i=0;i<label.length;i++){
            System.out.println(label[i]+"       "+val[i]);
        }
        System.out.println();
    }
}
