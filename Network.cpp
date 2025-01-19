#include "Network.h"
#include "iostream"
#include <cmath>

using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}


void Network::AddLayer(int numberOfNeurons)
{
    vector<Neuron> neurons;
    for(int i=0;i<numberOfNeurons;i++)
    {
        Neuron neuron = Neuron(i%2==0 ? 0 : 1,i%2==0 ? 0 : 1,i%2==0 ? 1 : 0);
        neurons.push_back(neuron);
    }
    layers.push_back(neurons);

    if(layers.size()>1)
    {
        vector<Neuron>& lastLayer = layers[layers.size()-2];
        vector<Neuron>& currentLayer = layers[layers.size()-1];

        for(int i=0;i<lastLayer.size();i++)
        {
            for(int j=0;j<numberOfNeurons;j++)
            {
                Weight weight = Weight(j%2==0 ? 0 : 1);
                lastLayer[i].addForwardWeight(weight);
                currentLayer[j].addBackwardWeight(weight);
            }
        }
    }
}

void Network::trainNetwork(vector<double> input)
{
    cout<<endl;
    vector<Neuron> &firstLayer = layers.front();
    for(int i=0;i<firstLayer.size();i++){
        (firstLayer[i]).setZValue(input[i]);
        (firstLayer[i]).setActivationValue(input[i]);
    }

    for(int i=1;i<layers.size();i++)
    {
        vector<Neuron> &currentLayer = layers[i];
        vector<Neuron> &prevLayer = layers[i-1];
        MatrixXd biasMatrix(currentLayer.size(),1);
        for(int j=0;j<currentLayer.size();j++)
        {
            biasMatrix(j,0) = currentLayer[j].getBias();
        }
        cout<<"Bias matrix is "<<biasMatrix<<"\n";
        MatrixXd weightMatrix(currentLayer.size(),prevLayer.size());
        for(int j=0;j<prevLayer.size();j++)
        {
            vector<Weight> weights = prevLayer.at(j).getForwardWeights();
            for(int k=0;k<weights.size();k++)
            {
                weightMatrix(k,j) = weights.at(k).getWeight();
            }
        }
        cout<<"Weight matrix is "<<weightMatrix<<"\n";


        MatrixXd activationValues(prevLayer.size(),1);

        for(int i=0;i<prevLayer.size();i++)
        {
            activationValues(i,0) = prevLayer[i].getActivationValue();
        }
        cout<<"\n";
        cout<<"Activation Values are: "<<activationValues<<endl;

        MatrixXd newActivationValues = (weightMatrix*activationValues)+biasMatrix;

        for(int i=0;i<currentLayer.size();i++)
        {
            currentLayer[i].setZValue(newActivationValues(i,0));
            currentLayer[i].setActivationValue(sigmoid(newActivationValues(i,0)));
        }

        cout<<"New Activation Values "<<newActivationValues<<endl;
    }
    


    cout<<"Training cycle completed"<<endl;
}

Network::Network(){

}