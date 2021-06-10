package com.firefly.ai;
import com.firefly.ai.listener.FitListener;
import com.firefly.ai.util.Sigmoid;

import java.util.Random;

public class Bp {
    private double rate;//学习率
    private int inputLayerLen;//输入层长度
    private int hideLayerLen;//隐藏层长度
    private int outLayerLen;//输出层长度

    private double[][] hideLayerW;//隐藏层的权重
    private double[] hideLayerB;//隐藏层的偏置
    private double[][] outLayerW;//输出层的权重
    private double[] outLayerB;//输出层的偏置

    public Bp(double rate, int inputLayerLen, int hideLayerLen, int outLayerLen) {
        this.rate = rate;
        this.inputLayerLen = inputLayerLen;
        this.hideLayerLen = hideLayerLen;
        this.outLayerLen = outLayerLen;

        hideLayerW=new double[inputLayerLen][hideLayerLen];
        hideLayerB=new double[hideLayerLen];
        outLayerW=new double[hideLayerLen][outLayerLen];
        outLayerB=new double[outLayerLen];

        randVals(hideLayerW);
        randVals(hideLayerB);
        randVals(outLayerW);
        randVals(outLayerB);
    }

    public void fit(double[][] input, double[][] label, int batch, int epoch, FitListener fitListener){
        int batchNum=input.length/batch;
        int batchIndex=0;
        for(int i=0;i<epoch;i++){
            if(batchIndex>=batchNum){
                batchIndex=0;
            }
            //隐藏层的值
            double[] hideLayerVal=new double[hideLayerLen];
            double[] outLayerVal=new double[outLayerLen];

            //输出层的偏导值
            double[] outLayerDe=new double[outLayerVal.length];
            //隐藏层的偏导值
            double[] hideLayerDe=new double[hideLayerVal.length];

            double[][] hideLayerWDiff=new double[inputLayerLen][hideLayerLen];
            double[] hideLayerBDiff=new double[hideLayerLen];
            double[][] outLayerWDiff=new double[hideLayerLen][outLayerLen];
            double[] outLayerBDiff=new double[outLayerLen];

            for(int row=batchIndex*batch;row<batchIndex*batch+batch;row++){
                //##################正向计算#########################
                //计算隐藏层的神经元值
                calcNeureLayer(input[row],hideLayerW,hideLayerB,hideLayerVal);
                //计算输出层的神经元值
                calcNeureLayer(hideLayerVal,outLayerW,outLayerB,outLayerVal);

                //##################反向更新#########################
                //计算label与输出层的误差函数1/2*(out-label)^2,  误差函数的偏导 out-label
                lossOutDe(outLayerVal,label[row],outLayerDe);
                //计算误差函数到隐藏层的偏导
                lossHideDe(outLayerDe,outLayerVal,outLayerW,hideLayerDe);

                //计算误差函数对输出层的b、w的偏导
                lossOutBDe(outLayerDe,outLayerVal,outLayerBDiff);
                lossOutWDe(outLayerDe,outLayerVal,hideLayerVal,outLayerWDiff);

                //计算误差函数对隐藏层的b、w的偏导
                lossHideBDe(hideLayerDe,hideLayerVal,input[row],hideLayerBDiff);
                lossHideWDe(hideLayerDe,hideLayerVal,input[row],hideLayerWDiff);
            }

            loss(input,label,fitListener);

            subVals(hideLayerB,hideLayerBDiff,rate);
            subVals(hideLayerW,hideLayerWDiff,rate);
            subVals(outLayerB,outLayerBDiff,rate);
            subVals(outLayerW,outLayerWDiff,rate);

            batchIndex++;
        }
    }

    public double[] predict(double[] input){
        //隐藏层的值
        double[] hideLayerVal=new double[hideLayerLen];
        double[] outLayerVal=new double[outLayerLen];

        //计算隐藏层的神经元值
        calcNeureLayer(input,hideLayerW,hideLayerB,hideLayerVal);
        //计算输出层的神经元值
        calcNeureLayer(hideLayerVal,outLayerW,outLayerB,outLayerVal);

        return outLayerVal;
    }

    private void loss(double[][] input, double[][] label,FitListener fitListener){
        if(fitListener!=null){
            double err=0;
            for(int row=0;row<input.length;row++){
                double[] val=predict(input[row]);
                for(int i=0;i<val.length;i++){
                    err+=Math.pow(label[row][i]-val[i],2);
                }
            }
            err*=0.5;
            fitListener.onLoss(err);
        }
    }

    private void subVals(double[] src,double[] add,double rate){
        for(int i=0;i<src.length;i++){
            src[i]-=rate*add[i];
        }
    }

    private void subVals(double[][] src,double[][] add,double rate){
        for(int i=0;i<src.length;i++){
            for(int j=0;j<src[i].length;j++){
                src[i][j]-=rate*add[i][j];
            }
        }
    }

    private void randVals(double[] data){
        for(int i=0;i<data.length;i++){
            data[i]=Math.random();
        }
    }

    private void randVals(double[][] data){
        Random random=new Random();
        for(int i=0;i<data.length;i++){
            for(int j=0;j<data[i].length;j++){
                data[i][j]=random.nextGaussian();//高斯分布
            }
        }
    }

    /**
     * 求误差函数对输出层的偏导
     * @return
     */
    private void lossOutDe(double[] out,double[] label, double[] outLayerDe){
        for(int i=0;i<out.length;i++){
            outLayerDe[i]=out[i]-label[i];
        }
    }

    /**
     * 误差函数对隐藏层的偏导
     * @param outLayerDe
     * @param outLayerVal
     * @param outLayerW
     * @param hideLayerDe
     */
    private void lossHideDe(double[] outLayerDe,double[] outLayerVal,double[][] outLayerW,double[] hideLayerDe){
        for(int j=0;j<hideLayerDe.length;j++){
            for(int i=0;i<outLayerVal.length;i++){
                hideLayerDe[j]+=outLayerDe[i]*Sigmoid.deByVal(outLayerVal[i])*outLayerW[j][i];
            }
        }
    }

    /**
     * 求误差函数对输出层B的偏导
     * @param outLayerDe
     * @param outLayerVal
     * @param outLayerBDiff
     */
    private void lossOutBDe(double[] outLayerDe,double[] outLayerVal,double[] outLayerBDiff){
        for(int i=0;i<outLayerDe.length;i++){
            outLayerBDiff[i]+=outLayerDe[i]*Sigmoid.deByVal(outLayerVal[i])*1;
        }
    }

    /**
     * 求误差函数对输出层W的偏导
     * @param outLayerDe
     * @param outLayerVal
     * @param outLayerWDiff
     * @param hideLayerVal
     */
    private void lossOutWDe(double[] outLayerDe,double[] outLayerVal,double[] hideLayerVal,double[][] outLayerWDiff){
        for(int i=0;i<outLayerWDiff.length;i++){
            for(int j=0;j<outLayerWDiff[i].length;j++){
                outLayerWDiff[i][j]+=outLayerDe[j]*Sigmoid.deByVal(outLayerVal[j])*hideLayerVal[i];
            }
        }
    }

    /**
     * 求误差函数对输出层B的偏导
     * @param hideLayerDe
     * @param hideLayerVal
     * @param inputLayerVal
     * @param hideLayerBDiff
     */
    private void lossHideBDe(double[] hideLayerDe,double[] hideLayerVal,double[] inputLayerVal,double[] hideLayerBDiff){
        for(int i=0;i<hideLayerBDiff.length;i++){
            hideLayerBDiff[i]+=hideLayerDe[i]*Sigmoid.deByVal(hideLayerVal[i])*1;
        }
    }

    /**
     * 求误差函数对输出层W的偏导
     * @param hideLayerDe
     * @param hideLayerVal
     * @param inputLayerVal
     * @param hideLayerWDiff
     */
    private void lossHideWDe(double[] hideLayerDe,double[] hideLayerVal,double[] inputLayerVal,double[][] hideLayerWDiff){
        for(int k=0;k<hideLayerWDiff.length;k++){
            for(int j=0;j<hideLayerWDiff[k].length;j++){
                hideLayerWDiff[k][j]+=hideLayerDe[j]*Sigmoid.deByVal(hideLayerVal[j])*inputLayerVal[k];
            }
        }
    }

    /**
     * 计算神经元层
     * @param input
     * @param w
     * @param b
     * @param out
     */
    private void calcNeureLayer(double[] input,double[][] w,double[] b,double[] out){
        for(int x=0;x<out.length;x++){
            //计算每个神经元的值
            out[x]=sigmoidWxb(input,w,b[x],true,x);
        }
    }

    /**
     * w*x+b
     * @param x
     * @param w
     * @param b
     * @return
     */
    private double wxb(double[] x,double[][] w,double b,boolean row,int index){
        double retVal=0;
        if(row){
            for(int i=0;i<x.length;i++){
                retVal+=x[i]*w[i][index];
            }
        }else{
            for(int i=0;i<x.length;i++){
                retVal+=x[i]*w[index][i];
            }
        }
        return retVal+b;
    }

    private double sigmoidWxb(double[] x,double[][] w,double b,boolean row,int index){
        return Sigmoid.calc(wxb(x,w,b,row,index));
    }
}
