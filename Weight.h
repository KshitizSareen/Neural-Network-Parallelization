#ifndef WEIGHT_H  // Include guard to prevent multiple inclusions
#define WEIGHT_H

class Weight{
    private:
        double weight;
        double changeInWeightOverCost;
    public:
        Weight(double weight);
        double getWeight();
        void setWeight(double weight);
        void setChangeInWeightOverCost(double cost);

};

#endif