#include "Weight.h"

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
