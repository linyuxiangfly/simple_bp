package com.firefly.ai.listener;

public interface FitListener {
    boolean isOnLoss(int epoch);
    void onLoss(int epoch,double loss);
}
