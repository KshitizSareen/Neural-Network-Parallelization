#ifndef NEURON_H  // Include guard to prevent multiple inclusions
#define NEURON_H
#include "Weight.h"
#include "vector"
using namespace std;

class Neuron{
    private:
        double activationValue;
        double zValue;
        double bias;
        double error;

        vector<Weight> forwardWeights;
        vector<Weight> backwardWeights;


    public:
        Neuron(double activationValue,double zValue,double bias);
        double getActivationValue();
        void setActivationValue(double value);
        double getZValue();
        void setZValue(double value);
        double getBias();
        void setBias(double bias);
        void setError(double error);
        double getError();

        void addForwardWeight(Weight &weight);

        vector<Weight> getForwardWeights();

        void addBackwardWeight(Weight &weight);

        vector<Weight> getBackwardWeights();
};

#endif