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
//        double[][] y={
//                {1,1},
//                {0,0},
//                {0,0},
//                {0,0},
//                {0,0},
//                {0,0},
//                {0,0},
//                {1,1},
//        };
        double[][] y={
                {1},
                {0},
                {0},
                {0},
                {0},
                {0},
                {0},
                {1},
        };

        Bp bp=new Bp(0.1,3,10,1);
        bp.fit(x, y, 1, 1000000, new FitListener() {
            public boolean isOnLoss(int epoch) {
                return epoch%10000==0;
            }

            public void onLoss(int epoch,double loss) {
                System.out.println(loss);
            }
        });

        double[][] hw=bp.getHideLayerW();
        double[] hb=bp.getHideLayerB();
        double[][] ow=bp.getOutLayerW();
        double[] ob=bp.getOutLayerB();

        printParam("HideLayerW",hw);
        System.out.println();
        printParam("HideLayerB",hb);
        System.out.println();
        printParam("OutLayerW",ow);
        System.out.println();
        printParam("OutLayerB",ob);
        System.out.println();

        for(int i=0;i<x.length;i++){
            print(y[i],bp.predict(x[i]));
        }
    }

    private static void printParam(String paramName,double[][] param){
        String retVal="";

        retVal=paramName+":\n[\n";
        for(int i=0;i<param.length;i++){
            retVal+="   [";
            for(int j=0;j<param[i].length;j++){
                retVal+=param[i][j]+(j==param[i].length-1?"":",");
            }
            retVal+=i==param.length-1?"]":"],\n";
        }
        retVal+="\n]";
        System.out.println(retVal);
    }

    private static void printParam(String paramName,double[] param){
        String retVal="";

        retVal=paramName+":\n[";
        for(int i=0;i<param.length;i++){
            retVal+=param[i]+(i==param.length-1?"":",");
        }
        retVal+="]";
        System.out.println(retVal);
    }

    private static void print(double[] label,double[] val){
        for(int i=0;i<label.length;i++){
            System.out.println(label[i]+"       "+val[i]);
        }
        System.out.println();
    }
}
