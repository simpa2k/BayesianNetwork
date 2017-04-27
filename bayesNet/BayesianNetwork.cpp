//
// Created by simon on 2017-04-26.
//

#include "armadillo"
#include "BayesianNetwork.h"

bool BayesianNetwork::add(std::string factorName) {
    return graph.add(factorName);
}

bool BayesianNetwork::record(std::string factor1, std::string factor2, int factor1State, int factor2State, double factor2Probability) {

    arma::mat* probabilities = graph.getWeight(factor1, factor2);

    if (probabilities == NULL) {

        arma::mat* values = new arma::mat(factor1State + 1, factor2State + 1, arma::fill::zeros); // making sure the matrix is at least as big as it has to be.
        values[factor1State, factor2State] = factor2Probability;

        return graph.connect(factor1, factor2, *values);

    }
    return true;

}

arma::mat BayesianNetwork::get(std::string hidden, std::map<std::string, int> visibleStates) {

    arma::mat currentStates = arma::mat(1, 1);

    for (auto const& it : visibleStates) {

        arma::mat* probabilities = graph.getWeight(hidden, it.first);

    }

    return currentStates;
}

