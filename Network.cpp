#include "Network.h"
#include "iostream"
#include <cmath>
#include "memory"

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

double Network::calculateErrorForLastNeuron(shared_ptr<Neuron> neuron,double outputValue,int numberOfOutputs)
{
    double sigmoidDerivative = sigmoidDerivativeValue(neuron->getZValue());
    double costDerivative = costDerivativeValue(neuron->getActivationValue(),outputValue,numberOfOutputs);
    return sigmoidDerivative*costDerivative;
}

double Network::calculateErrorForHiddenLayer(shared_ptr<Neuron> neuron)
{
    double sigmoidDerivative = sigmoidDerivativeValue(neuron->getZValue());
    double sumErrorOfNextLayer = 0;
    vector<shared_ptr<Weight>> weights = neuron->getForwardWeights();
    for(int i=0;i<weights.size();i++)
    {
        shared_ptr<Weight> weight = weights[i];
        int neuronLayer = weight->getNextNeuronLayer();
        int neuronIndex = weight->getNextNeuronIndex();
        shared_ptr<Neuron> neuron = layers[neuronLayer][neuronIndex];
        sumErrorOfNextLayer+=neuron->getError()*weight->getWeight();
    }
    return sigmoidDerivative*sumErrorOfNextLayer;

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
            }
        }
    }
}

void Network::initializeLayerInput(const vector<double>& input) {
    vector<shared_ptr<Neuron>> firstLayer = layers.front();
    for (size_t i = 0; i < firstLayer.size(); ++i) {
        firstLayer[i]->setZValue(input[i]);
        firstLayer[i]->setActivationValue(input[i]);
    }
}

MatrixXd Network::calculateWeightMatrix(const vector<shared_ptr<Neuron>>& prevLayer) {
    MatrixXd weightMatrix(prevLayer.front()->getForwardWeights().size(), prevLayer.size());
    for (size_t j = 0; j < prevLayer.size(); ++j) {
        vector<shared_ptr<Weight>> weights = prevLayer[j]->getForwardWeights();
        for (size_t k = 0; k < weights.size(); ++k) {
            weightMatrix(k, j) = weights[k]->getWeight();
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
        
        MatrixXd weightMatrix = calculateWeightMatrix(prevLayer);
        MatrixXd biasMatrix = calculateBiasMatrix(currentLayer);
        MatrixXd activationValues = getActivationValues(prevLayer);
        MatrixXd newActivationValues = (weightMatrix * activationValues) + biasMatrix;

        for (size_t j = 0; j < currentLayer.size(); ++j) {
            currentLayer[j]->setZValue(newActivationValues(j, 0));
            currentLayer[j]->setActivationValue(sigmoid(newActivationValues(j, 0)));
        }
    }
}

void Network::backwardPropagate(double outputValue) {
    // Handle the last layer separately
    vector<shared_ptr<Neuron>> backLayer = layers.back();
    for (size_t i = 0; i < backLayer.size(); ++i) {
        shared_ptr<Neuron> neuron = backLayer[i];
        double error = calculateErrorForLastNeuron(neuron, outputValue == i ? 1 : 0, backLayer.size());
        neuron->setError(error);
        adjustWeightsAndBiases(neuron, error);
    }

    // Handle hidden layers
    for (int i = layers.size() - 2; i >= 1; --i) {
        vector<shared_ptr<Neuron>> currentLayer = layers[i];
        for (shared_ptr<Neuron>& neuron : currentLayer) {
            double error = calculateErrorForHiddenLayer(neuron);
            neuron->setError(error);
            adjustWeightsAndBiases(neuron, error);
        }
    }
}

void Network::adjustWeightsAndBiases(shared_ptr<Neuron>& neuron, double error) {
    vector<shared_ptr<Weight>> backwardWeights = neuron->getBackwardWeights();
    for (shared_ptr<Weight>& weight : backwardWeights) {
        adjustWeight(error, weight, learningRate);
    }
    adjustBias(error, neuron);
}

void Network::updateWeightsAndBiases() {
    for (vector<shared_ptr<Neuron>>& layer : layers) {
        for (shared_ptr<Neuron>& neuron : layer) {
            vector<shared_ptr<Weight>> forwardWeights = neuron->getForwardWeights();
            for (shared_ptr<Weight>& weight : forwardWeights) {
                weight->setWeight(weight->getWeight() - weight->getChangeInWeight());
            }
            neuron->setBias(neuron->getBias() - neuron->getChangeInBias());
        }
    }
}

void Network::trainNetwork(vector<double> input, double outputValue) {
    initializeLayerInput(input);
    forwardPropagate();
    backwardPropagate(outputValue);
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

Network::Network(){
    
}