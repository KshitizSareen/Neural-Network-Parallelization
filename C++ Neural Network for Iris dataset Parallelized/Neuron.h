#ifndef NEURON_H  // Include guard to prevent multiple inclusions
#define NEURON_H
#include "Weight.h"
#include "vector"
#include "memory"
using namespace std;

class Neuron{
    private:
        double activationValue;
        double zValue;
        double bias;
        double error = 0;
        double changeInBias;

        vector<shared_ptr<Weight>> forwardWeights;
        vector<shared_ptr<Weight>> backwardWeights;


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
        double getChangeInBias();
        void setChangeInBias(double bias);

        void addForwardWeight(shared_ptr<Weight> weight);

        vector<shared_ptr<Weight>> getForwardWeights();

        void addBackwardWeight(shared_ptr<Weight>weight);

        vector<shared_ptr<Weight>> getBackwardWeights();
};

#endif