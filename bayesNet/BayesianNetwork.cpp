//
// Created by simon on 2017-04-26.
//

#include "armadillo"
#include "BayesianNetwork.h"

const arma::uword NUM_STATES = 2; // Solve this in a more elegant way.

bool BayesianNetwork::add(std::string factorName) {
    return graph.add(factorName);
}

bool BayesianNetwork::record(std::string factor1, std::string factor2, arma::uword factor1State, arma::uword factor2State, double factor2Probability) {

    arma::mat values;
    arma::mat* probabilities = graph.getWeight(factor1, factor2);

    if (probabilities != NULL) {
        values = *probabilities;
    } else {
        values = arma::mat(NUM_STATES, NUM_STATES, arma::fill::zeros);
    }

    values(factor2State, factor1State) = factor2Probability;

    bool result = graph.connect(factor1, factor2, values);
    delete probabilities;

    return result;

}

arma::mat BayesianNetwork::get(std::string hidden, std::map<std::string, arma::uword> visibleStates) {

    arma::mat currentStates;

    for (auto const& it : visibleStates) {

        arma::mat* probabilities = graph.getWeight(hidden, it.first);
        currentStates = arma::join_cols(currentStates, probabilities->row(it.second));

        delete probabilities;

    }

    return currentStates;

}
