package com.firefly.ai.util;

public class Sigmoid {
    /**
     * sigmoid函数
     * @param x
     * @return
     */
    public static double calc(double x){
        return 1/(1+Math.exp(-x));
    }

    /**
     * sigmoid导数函数
     * @param x
     * @return
     */
    public static double de(double x){
        double s=calc(x);
        return s*(1-s);
    }

    /**
     * 由sigmoid的函数值来计算偏导
     * @param val
     * @return
     */
    public static double deByVal(double val){
        return val*(1-val);
    }
}
