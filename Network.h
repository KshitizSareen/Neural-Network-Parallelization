#ifndef NETWORK_H  // Include guard to prevent multiple inclusions
#define NETWORK_H
#include "vector"
#include "Neuron.h"
#include "Eigen/Dense"
using namespace Eigen;
using namespace std;

class Network{
    private:
        vector<vector<Neuron>> layers;
        double learningRate;
    public:
        Network();
        void AddLayer(int numberOfNeurons);

        void trainNetwork(vector<double> input,double outputValue);

        double sigmoidDerivativeValue(double input);

        double costDerivativeValue(double predictedValue,double outputValue,int numberOfOutputs);

        double calculateErrorForLastNeuron(Neuron neuron,double outputValue,int numberOfOutputs);

        void adjustWeight(double errorValue,Weight &weight,double learningRate);

        double getLearningRate();

        void setLearningRate(double learningRate);

        double calculateErrorForHiddenLayer(Neuron neuron);
};

#endif