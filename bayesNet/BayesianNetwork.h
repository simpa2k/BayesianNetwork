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
    bool add(std::string);
    bool record(std::string, std::string, arma::uword, arma::uword, double);
    arma::mat get(std::string, std::map<std::string, arma::uword>);
    std::vector<int> simulateHiddenData(const std::vector<double>, const int);
    std::map<std::string, std::vector<int>> simulateVisibleData(const std::string, const std::vector<int>, const int);

};


#endif //GRAPH_BAYESIANNETWORK_H
