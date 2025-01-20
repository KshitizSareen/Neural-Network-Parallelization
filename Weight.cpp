#include "Weight.h"
#include "Neuron.h"

Weight::Weight( double weight)
{
    this->weight = weight;
}

double Weight::getWeight(){
    return weight;
}

void Weight::setWeight(double weight)
{
    this->weight = weight;
}


void Weight::setChangeInWeight(double change)
{
    this->changeInWeight = change;
}

void Weight::setPrevNeuronLayer(int prevNeuronLayer){
    this->prevNeuronLayer = prevNeuronLayer;
}

void Weight::setNextNeuronLayer(int nextNeuronLayer){
    this->nextNeuronLayer = nextNeuronLayer;
}

void Weight::setPrevNeuronIndex(int prevNeuronIndex){
    this->prevNeuronIndex = prevNeuronIndex;
}

void Weight::setNextNeuronIndex(int nextNeuronIndex){
    this->nextNeuronIndex = nextNeuronIndex;
}

// Getters
int Weight::getPrevNeuronLayer()  {
    return prevNeuronLayer;
}

int Weight::getNextNeuronLayer()  {
    return nextNeuronLayer;
}

int Weight::getPrevNeuronIndex() {
    return prevNeuronIndex;
}

int Weight::getNextNeuronIndex() {
    return nextNeuronIndex;
}

double Weight::getChangeInWeight()
{
    return changeInWeight;
}
