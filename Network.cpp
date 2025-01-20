#include "Network.h"
#include "iostream"
#include <cmath>

using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double Network::sigmoidDerivativeValue(double input)
{
    return sigmoid(input)*(1-sigmoid(input));
}

double Network::costDerivativeValue(double predictedValue,double outputValue,int numberOfOutputs)
{
    cout<<"Predicted Value is "<<predictedValue<<" ,Output Value is "<<outputValue<<" ,number of outputs is "<<numberOfOutputs<<"\n";
    return (predictedValue-outputValue)/numberOfOutputs;
}

double Network::calculateErrorForLastNeuron(Neuron neuron,double outputValue,int numberOfOutputs)
{
    double sigmoidDerivative = sigmoidDerivativeValue(neuron.getZValue());
    cout<<"Sigmoid value is "<<sigmoidDerivative<<"\n";
    cout<<"Activation value is "<<neuron.getActivationValue()<<"\n";
    double costDerivative = costDerivativeValue(neuron.getActivationValue(),outputValue,numberOfOutputs);
    cout<<"Cost value is "<<costDerivative<<"\n";
    cout<<"Error value is "<<sigmoidDerivative*costDerivative<<"\n";
    return sigmoidDerivative*costDerivative;
}

double Network::calculateErrorForHiddenLayer(Neuron neuron)
{
    double sigmoidDerivative = sigmoidDerivativeValue(neuron.getZValue());
    double sumErrorOfNextLayer = 0;
    vector<Weight> weights = neuron.getForwardWeights();
    for(int i=0;i<neuron.getForwardWeights().size();i++)
    {
        Weight weight = weights[i];
        int neuronLayer = weight.getNextNeuronLayer();
        int neuronIndex = weight.getNextNeuronIndex();
        Neuron neuron = layers[neuronLayer][neuronIndex];
        sumErrorOfNextLayer+=neuron.getError()*weight.getWeight();
    }
    return sigmoidDerivative*sumErrorOfNextLayer;

}

void Network::adjustWeight(double error,Weight &weight,double learningRate)
{
        int neuronLayer = weight.getPrevNeuronLayer();
        int neuronIndex = weight.getPrevNeuronIndex();
        Neuron neuron = layers[neuronLayer][neuronIndex];
        double changeInCostOverWeight = error*neuron.getActivationValue();
        cout<<"Error is adjust weight is "<<error<<"\n";
        cout<<"Activation is adjust weight is "<<neuron.getActivationValue()<<"\n";
        cout<<"Change in cost over weight is adjust weight is "<<changeInCostOverWeight<<"\n";
        cout<<"Change is adjust weight is "<<learningRate*changeInCostOverWeight<<"\n";
        weight.setChangeInWeight(learningRate*changeInCostOverWeight);
}

double Network::getLearningRate()  {
    return learningRate; // Return the value of the private variable
}

void Network::setLearningRate(double learningRate) {
    this->learningRate = learningRate; // Assign value to private variable
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
                weight.setPrevNeuronLayer(layers.size()-2);
                weight.setNextNeuronLayer(layers.size()-1);
                weight.setPrevNeuronIndex(i);
                weight.setNextNeuronIndex(j);
                lastLayer[i].addForwardWeight(weight);
                currentLayer[j].addBackwardWeight(weight);
            }
        }
    }
}

void Network::trainNetwork(vector<double> input,double outputValue)
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

    vector<Neuron> &backLayer = layers.back();
    cout<<"Errors for last layer are"<<"\n";
    for(int i=0;i<backLayer.size();i++){
        Neuron& neuron = backLayer[i];
        double error = calculateErrorForLastNeuron(neuron,outputValue == i ? 1 : 0,backLayer.size());
        neuron.setError(error);
        cout<<neuron.getError()<<" ";
        vector<Weight> backwardWeights = neuron.getBackwardWeights();
        for(int j=0;j<backwardWeights.size();j++)
        {
            Weight &weight = backwardWeights[j];
            adjustWeight(error,weight,learningRate);
            cout<<"Change in weight is "<<weight.getChangeInWeight()<<"\n";
        }
    }
    cout<<"\n";

/*
    for(int i=layers.size()-2;i>=1;i--)
    {
        vector<Neuron> &currentLayer = layers[i];
        for(int j=0;j<currentLayer.size();j++)
        {
            Neuron& neuron = currentLayer[j];
            double error = calculateErrorForHiddenLayer(neuron);
            vector<Weight> backwardWeights = neuron.getBackwardWeights();
            for(int j=0;j<backwardWeights.size();j++)
            {
                Weight weight = backwardWeights[j];
                adjustWeight(error,weight,learningRate);
            }
        }
    }
    */

    for(int i=layers.size()-2;i<layers.size()-1;i++)
    {
        cout<<"Weights for layers "<<i<<" are \n";
        vector<Neuron> &currentLayer = layers[i];
        for(int j=0;j<currentLayer.size();j++)
        {
            Neuron& neuron = currentLayer[j];
            vector<Weight> forwardWeights = neuron.getForwardWeights();
            for(int j=0;j<forwardWeights.size();j++)
            {
                Weight &weight = forwardWeights[j];
                weight.setWeight(weight.getWeight()-weight.getChangeInWeight());
                cout<<"Change is weight is "<<weight.getChangeInWeight()<<" "<<"New weight is "<<weight.getWeight()<<"\n";
            }
        }
        cout<<endl;
    }



    


    cout<<"Training cycle completed"<<endl;
}



Network::Network(){

}