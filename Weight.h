#ifndef WEIGHT_H  // Include guard to prevent multiple inclusions
#define WEIGHT_H

class Weight{
    private:
        double weight;
        double changeInWeight;
        int prevNeuronLayer;
        int nextNeuronLayer;
        int prevNeuronIndex;
        int nextNeuronIndex;

    public:
        Weight(double weight);
        double getWeight();
        void setWeight(double weight);
        void setChangeInWeight(double change);
        void setPrevNeuronLayer(int prevNeuronLayer);
        void setNextNeuronLayer(int nextNeuronLayer);
        void setPrevNeuronIndex(int prevNeuronLayer);
        void setNextNeuronIndex(int nextNeuronLayer);
        int getPrevNeuronLayer();
        int getNextNeuronLayer();
        int getPrevNeuronIndex();
        int getNextNeuronIndex();
        double getChangeInWeight();
};

#endif