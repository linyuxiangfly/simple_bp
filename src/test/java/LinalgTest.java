import com.firefly.ai.util.Linalg;

public class LinalgTest {
    public static void main(String[] args){
        double a1[]={1,2,3};
        double a2[]={1,2,3,4,5};
        double out1[]=new double[5];
        double out2[]=new double[3];

        double b[][]={
                {1,2,3,4,5},
                {6,7,8,9,10},
                {11,12,13,14,15}
        };

        System.out.println(1*5+2*10+3*15);

        Linalg.inner(a1,b,out1,true);
        Linalg.inner(a2,b,out2,false);

        for(int i=0;i<out1.length;i++){
            System.out.println("out1["+i+"]="+out1[i]);
        }

        for(int i=0;i<out2.length;i++){
            System.out.println("out2["+i+"]="+out2[i]);
        }
    }
}
