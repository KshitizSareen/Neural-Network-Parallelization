#ifndef NEURON_H  // Include guard to prevent multiple inclusions
#define NEURON_H
#include "Weight.h"
#include "vector"
using namespace std;

class Neuron{
    private:
        double value;
        double bias;

        vector<Weight> weights;


    public:
        Neuron(double value,double bias);
        double getValue();
        void setValue(double value);
        double getBias();
        void setBias(double bias);

        void addWeight(Weight weight);

        vector<Weight> getWeights();
};

#endif