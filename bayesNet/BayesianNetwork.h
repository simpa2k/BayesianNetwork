//
// Created by simon on 2017-04-26.
//

#ifndef GRAPH_BAYESIANNETWORK_H
#define GRAPH_BAYESIANNETWORK_H

#include <armadillo>
#include "../directedGraph/Graph.h"
#include "brain/Brain.h"

class BayesianNetwork {

    Graph<std::string, arma::mat> graph;
    Brain brain = Brain(400);
    arma::uword numStates = 2;

public:
    BayesianNetwork();
    BayesianNetwork(arma::uword);

    arma::uword getNumStates() const;

    bool add(std::string);
    bool record(std::string, std::string, arma::uword, arma::uword, double);

    arma::mat get(std::string, std::map<std::string, arma::uword>);

    arma::rowvec simulateHiddenData(std::vector<double>, int);
    std::map<std::string, std::vector<int>> simulateVisibleData(std::string, const arma::rowvec, int);

    arma::rowvec computeThetaHidden(arma::rowvec dataHidden);
    arma::mat computeThetaVisible(arma::rowvec dataHidden, std::map<std::string, std::vector<int>> dataVisible);

};


#endif //GRAPH_BAYESIANNETWORK_H
