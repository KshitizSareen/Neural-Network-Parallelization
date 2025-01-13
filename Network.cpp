#include "Network.h"
#include "iostream"
using namespace std;

void Network::AddLayer(int numberOfNeurons)
{
    vector<Neuron> neurons;
    for(int i=0;i<numberOfNeurons;i++)
    {
        Neuron neuron = Neuron(2,2);
        neurons.push_back(neuron);
    }
    layers.push_back(neurons);

    if(layers.size()>1)
    {
        vector<Neuron> lastLayer = layers[layers.size()-2];
        vector<Neuron> currentLayer = layers[layers.size()-1];

        for(int i=0;i<lastLayer.size();i++)
        {
            for(int j=0;j<numberOfNeurons;j++)
            {
                Weight weight = Weight(2);
                lastLayer[i].addWeight(weight);
                cout<<"Added weight to make size "<<lastLayer[i].getWeights().size()<<"\n";
            }
        }
    }
}

void Network::trainNetwork(vector<double> input)
{
    cout<<endl;
    vector<Neuron> &firstLayer = layers.front();
    for(int i=0;i<firstLayer.size();i++){
        (firstLayer[i]).setValue(input[i]);
    }
    cout<<"First layer value is: ";
    for(int i=0;i<firstLayer.size();i++){
        cout<<(firstLayer[i]).getValue()<<" ";
    }
    cout<<endl;

    cout<<"Weights size:"<<layers[0][0].getWeights().size()<<"\n";

    for(int i=1;i<layers.size();i++)
    {
        vector<Neuron> &currentLayer = layers[i];
        vector<Neuron> &prevLayer = layers[i-1];
        cout<<"Prev layer value is: ";
        for(int j=0;j<prevLayer.size();j++){
            cout<<prevLayer[j].getValue()<<" ";
        }
        MatrixXd biasMatrix(currentLayer.size(),1);
        for(int j=0;j<currentLayer.size();j++)
        {
            biasMatrix(j,0) = currentLayer[j].getBias();
        }
        cout<<"Bias matrix is "<<biasMatrix<<"\n";
        MatrixXd weightMatrix(currentLayer.size(),prevLayer.size());
        for(int j=0;j<prevLayer.size();j++)
        {
            vector<Weight> weights = prevLayer.at(j).getWeights();
            cout<<"Weight size is "<<prevLayer.at(j).getWeights().size()<<endl;
            for(int k=0;k<weights.size();k++)
            {
                weightMatrix(j,k) = weights.at(k).getWeight();
                cout<<"Weight is "<<weights.at(k).getWeight()<<"\n";
            }
        }
        cout<<"Weight matrix is "<<weightMatrix<<"\n";


        MatrixXd activationValues(prevLayer.size(),1);

        for(int i=0;i<prevLayer.size();i++)
        {
            activationValues(i,0) = prevLayer[i].getValue();
        }
        cout<<"\n";
        cout<<"Activation Values are: "<<activationValues<<endl;

        MatrixXd newActivationValues = (weightMatrix*activationValues)+biasMatrix;

        for(int i=0;i<currentLayer.size();i++)
        {
            currentLayer[i].setValue(newActivationValues(i,0));
        }

        cout<<"New Activation Values "<<newActivationValues<<endl;
    }

    cout<<"Training cycle completed"<<endl;
}

Network::Network(){

}