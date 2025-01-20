#include "Neuron.h"
#include "iostream"

using namespace std;

Neuron::Neuron(double activationValue,double zValue,double bias){
    this->activationValue = activationValue;
    this->zValue = zValue;
    this->bias = bias;
}

double Neuron::getActivationValue(){
    return activationValue;
}

void Neuron::setActivationValue(double value){
    this->activationValue = value;

}

double Neuron::getZValue(){
    return zValue;
}

void Neuron::setZValue(double value){
    this->zValue = value;

}

double Neuron::getBias(){
    return bias;
}

void Neuron::setBias(double bias)
{
    this->bias = bias;
}

void Neuron::addForwardWeight(Weight &weight)
{
    forwardWeights.push_back(weight);
}

void Neuron::addBackwardWeight(Weight &weight)
{
    backwardWeights.push_back(weight);
}

vector<Weight> Neuron::getForwardWeights(){
    return forwardWeights;
}

vector<Weight> Neuron::getBackwardWeights(){
    return backwardWeights;
}

void Neuron::setError(double error){
    this->error = error;
}

double Neuron::getError(){
    return this->error;
}
