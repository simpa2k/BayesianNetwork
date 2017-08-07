//
// Created by simon on 2017-04-26.
//

#ifndef GRAPH_BAYESIANNETWORK_H
#define GRAPH_BAYESIANNETWORK_H

#include <armadillo>
#include "../directedGraph/Graph.h"
#include "brain/Brain.h"
#include <ctime>

/**
 * Class representing a Bayesian network. Based on a
 * directed, acyclic graph.
 */
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
    bool record(std::string, std::string, arma::uword, arma::uword);
    bool erase(std::string, std::string, arma::uword, arma::uword);

    arma::mat get(std::string, std::map<std::string, arma::uword>);

    arma::rowvec simulateHiddenData(std::vector<double>, int);
    std::map<std::string, arma::rowvec> simulateVisibleData(std::string, arma::rowvec, int);
    std::map<std::string, arma::rowvec> simulateVisibleData(std::map<std::string, arma::mat>, std::string, arma::rowvec, int);

    arma::rowvec computeThetaHidden(arma::rowvec dataHidden);

    std::map<std::string, arma::mat> computeThetaVisible(arma::rowvec dataHidden, std::map<std::string, arma::rowvec> dataVisible);
    std::map<std::string, arma::mat> computeThetaVisible(std::string);

};


#endif //GRAPH_BAYESIANNETWORK_H
