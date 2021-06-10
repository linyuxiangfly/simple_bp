package com.firefly.ai.util;

public class Linalg {
    /**
     * 计算内积
     * @param a
     * @param b
     * @param out 输出结果的数组
     * @param row 是按行来计算还是按列来计算
     */
    public static void inner(double[] a,double[][] b,double[] out,boolean row){
        for(int x=0;x<out.length;x++){
            out[x]=inner(a,b,row,x);
        }
    }

    /**
     * 计算内积
     * @param a
     * @param b
     * @param row 是按行来计算还是按列来计算
     * @param index 如果row=true说明是指定第几列数，如果row=false说明这里是指定第几行数
     * @return
     */
    public static double inner(double[] a,double[][] b,boolean row,int index){
        double retVal=0;
        if(row){
            for(int i=0;i<a.length;i++){
                retVal+=a[i]*b[i][index];
            }
        }else{
            for(int i=0;i<a.length;i++){
                retVal+=a[i]*b[index][i];
            }
        }
        return retVal;
    }
}
