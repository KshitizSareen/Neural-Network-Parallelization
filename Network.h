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
    public:
        Network();
        void AddLayer(int numberOfNeurons);

        void trainNetwork(vector<double> input);
};

#endif