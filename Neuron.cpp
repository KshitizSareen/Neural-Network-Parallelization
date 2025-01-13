#include "Neuron.h"
#include "iostream"

using namespace std;

Neuron::Neuron(double value,double bias){
    this->value = value;
    this->bias = bias;
}

double Neuron::getValue(){
    return value;
}

void Neuron::setValue(double value){
    this->value = value;

    cout<<"New value is "<<this->value<<endl;
}

double Neuron::getBias(){
    return bias;
}

void Neuron::setBias(double bias)
{
    this->bias = bias;
}

void Neuron::addWeight(Weight weight)
{
    weights.push_back(weight);
}

vector<Weight> Neuron::getWeights(){
    return weights;
}

