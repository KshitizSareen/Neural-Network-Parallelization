#include "Network.h"
#include "iostream"
#include <cmath>
#include "memory"

#include "omp.h"
#include "algorithm"

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
    return (predictedValue-outputValue)/numberOfOutputs;
}


void Network::adjustWeight(double error,shared_ptr<Weight> weight,double learningRate)
{
        int neuronLayer = weight->getPrevNeuronLayer();
        int neuronIndex = weight->getPrevNeuronIndex();
        shared_ptr<Neuron> neuron = layers[neuronLayer][neuronIndex];
        double changeInCostOverWeight = error*neuron->getActivationValue();
        weight->setChangeInWeight(learningRate*changeInCostOverWeight);
}

void Network::adjustBias(double error,shared_ptr<Neuron> neuron)
{
        double changeInCostOverBias = error;
        neuron->setChangeInBias(learningRate*changeInCostOverBias);
}

double Network::getLearningRate()  {
    return learningRate; // Return the value of the private variable
}

void Network::setLearningRate(double learningRate) {
    this->learningRate = learningRate; // Assign value to private variable
}


void Network::AddLayer(int numberOfNeurons)
{
    vector<shared_ptr<Neuron>> neurons;
    for(int i=0;i<numberOfNeurons;i++)
    {
        shared_ptr<Neuron> neuron = make_shared<Neuron>(i%2==0 ? 0 : 1,i%2==0 ? 0 : 1,0.5);
        neurons.push_back(neuron);
    }
    layers.push_back(neurons);

    if(layers.size()>1)
    {
        vector<shared_ptr<Neuron>> lastLayer = layers[layers.size()-2];
        vector<shared_ptr<Neuron>> currentLayer = layers[layers.size()-1];

        for(int i=0;i<lastLayer.size();i++)
        {
            for(int j=0;j<numberOfNeurons;j++)
            {
                shared_ptr<Weight> weight = make_shared<Weight>(0.5);
                weight->setPrevNeuronLayer(layers.size()-2);
                weight->setNextNeuronLayer(layers.size()-1);
                weight->setPrevNeuronIndex(i);
                weight->setNextNeuronIndex(j);
                lastLayer[i]->addForwardWeight(weight);
                currentLayer[j]->addBackwardWeight(weight);
                weights.push_back(weight);
            }
        }
    }
}


void Network::initializeLayerInput(const vector<double>& input) {
    vector<shared_ptr<Neuron>> firstLayer = layers.front();

        for (size_t i =0; i < input.size(); ++i) {
            firstLayer[i]->setZValue(input[i]);
            firstLayer[i]->setActivationValue(input[i]);
        }
}

MatrixXd Network::calculateWeightMatrix(const vector<shared_ptr<Neuron>>& prevLayer) {
    size_t rows = prevLayer.front()->getForwardWeights().size();
    size_t cols = prevLayer.size();

    MatrixXd weightMatrix(rows, cols);

    for(int i=0;i<prevLayer.size();i++)
    {
        vector<shared_ptr<Weight>> weights = prevLayer[i]->getForwardWeights();
        for(int j=0;j<weights.size();j++)
        {
            shared_ptr<Weight> weight = weights[j];
            weightMatrix(j,i) = weight->getWeight();
        }
    }

    return weightMatrix;
}


MatrixXd Network::calculateBiasMatrix(const vector<shared_ptr<Neuron>>& currentLayer) {
    MatrixXd biasMatrix(currentLayer.size(), 1);

    for (size_t j = 0; j < currentLayer.size(); ++j) {
        biasMatrix(j, 0) = currentLayer[j]->getBias();
    }
    
    return biasMatrix;
}

MatrixXd Network::getActivationValues(const vector<shared_ptr<Neuron>>& layer) {
    MatrixXd activationValues(layer.size(), 1);
    
    for (size_t i = 0; i < layer.size(); ++i) {
        activationValues(i, 0) = layer[i]->getActivationValue();
    }
    
    return activationValues;
}
void Network::forwardPropagate() {
    for (size_t i = 1; i < layers.size(); ++i) {
        vector<shared_ptr<Neuron>> currentLayer = layers[i];
        vector<shared_ptr<Neuron>> prevLayer = layers[i - 1];
        MatrixXd weightMatrix;
        MatrixXd biasMatrix;
        MatrixXd activationValues;

        MatrixXd newActivationValues;
        weightMatrix = calculateWeightMatrix(prevLayer);
        
        biasMatrix = calculateBiasMatrix(currentLayer);
    
        activationValues = getActivationValues(prevLayer);
        newActivationValues = (weightMatrix * activationValues) + biasMatrix;



        for (size_t j = 0; j < currentLayer.size(); ++j) {
            double zValue = newActivationValues(j, 0);
            currentLayer[j]->setZValue(zValue);
            currentLayer[j]->setActivationValue(sigmoid(zValue));
        }
    }
}

void Network::calculateErrorForLastLayer(vector<shared_ptr<Neuron>> neurons, int numberOfOutputs, int outputValue)
{
    for (size_t i = 0; i < neurons.size(); ++i) {
        shared_ptr<Neuron> neuron = neurons[i];
        double sigmoidDerivative = sigmoidDerivativeValue(neuron->getZValue());
        double costDerivative = costDerivativeValue(neuron->getActivationValue(), outputValue == i ? 1 : 0, numberOfOutputs);
        neuron->setError(sigmoidDerivative * costDerivative);
    }
}

void Network::calculateErrorForHiddenLayer(vector<shared_ptr<Neuron>> neurons)
{
    for (size_t i = 0; i < neurons.size(); ++i) {
        shared_ptr<Neuron> neuron = neurons[i];
        double sigmoidDerivative = sigmoidDerivativeValue(neuron->getZValue());
        double sumErrorOfNextLayer = 0;

        vector<shared_ptr<Weight>> weights = neuron->getForwardWeights();
        for (size_t j = 0; j < weights.size(); ++j) {
            shared_ptr<Weight> weight = weights[j];
            int neuronLayer = weight->getNextNeuronLayer();
            int neuronIndex = weight->getNextNeuronIndex();
            shared_ptr<Neuron> nextNeuron = layers[neuronLayer][neuronIndex];
            sumErrorOfNextLayer += nextNeuron->getError() * weight->getWeight();
        }

        neuron->setError(sigmoidDerivative * sumErrorOfNextLayer);
    }
}

void Network::backwardPropagate(double outputValue) {
    // Handle the last layer separately

    vector<shared_ptr<Neuron>> backLayer = layers.back();
    calculateErrorForLastLayer(backLayer,backLayer.size(),outputValue);

    // Handle hidden layers
    for (int i = layers.size() - 2; i >= 1; --i) {
        calculateErrorForHiddenLayer(layers[i]);
        }
}

void Network::adjustWeightsAndBiases() {
    for (int i = 0; i < layers.size(); i++) {
        vector<shared_ptr<Neuron>> currentLayer = layers[i];

        size_t neuronsSize = currentLayer.size();
        size_t weightsSize = currentLayer[0]->getBackwardWeights().size();
        size_t totalConnections = neuronsSize * weightsSize;

        for (size_t index = 0;index < totalConnections; ++index) {
            int i = index / weightsSize; // Neuron index
            int j = index % weightsSize; // Weight index
            shared_ptr<Neuron> neuron = currentLayer[i];
            shared_ptr<Weight> weight = neuron->getBackwardWeights()[j];
            double error = neuron->getError();


            weight->setChangeInWeight(learningRate * error * layers[weight->getPrevNeuronLayer()][weight->getPrevNeuronIndex()]->getActivationValue());

            neuron->setChangeInBias(learningRate * error);
        }
    }
}

void Network::updateWeightsAndBiases() {

    for (int i = 0; i < layers.size(); i++) {
        vector<shared_ptr<Neuron>> currentLayer = layers[i];

        size_t neuronsSize = currentLayer.size();

        for (size_t index = 0;index < neuronsSize; ++index) {
            shared_ptr<Neuron> neuron = currentLayer[index];
            neuron->setBias(neuron->getBias() - neuron->getChangeInBias());
        }

    }

    for (size_t index = 0;index < weights.size(); ++index) {
            shared_ptr<Weight> weight = weights[index];
            weight->setWeight(weight->getWeight()-weight->getChangeInWeight());
        }
}

void Network::trainNetwork(vector<double> input, double outputValue) {
    initializeLayerInput(input);
    forwardPropagate();
    backwardPropagate(outputValue);
    adjustWeightsAndBiases();
    updateWeightsAndBiases();
}

double Network::testNetwork(vector<double> input, double outputValue) {
    initializeLayerInput(input);
    forwardPropagate();

    vector<shared_ptr<Neuron>> backLayer = layers.back();
    double totalError = 0.0;
    for (size_t i = 0; i < backLayer.size(); ++i) {
        double error = pow((backLayer[i]->getActivationValue() - (outputValue == i ? 1 : 0)), 2);
        totalError += error;
    }
    return (1.0 / (2.0 * backLayer.size())) * totalError;
}

vector<double> Network::getOutput(vector<double> input) {
    initializeLayerInput(input);
    forwardPropagate();

    vector<shared_ptr<Neuron>> backLayer = layers.back();
    vector<double> output;
    for (const shared_ptr<Neuron>& neuron : backLayer) {
        output.push_back(neuron->getActivationValue());
    }
    return output;
}

