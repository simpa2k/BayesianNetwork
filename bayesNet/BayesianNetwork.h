//
// Created by simon on 2017-04-26.
//

#ifndef GRAPH_BAYESIANNETWORK_H
#define GRAPH_BAYESIANNETWORK_H

#include <armadillo>
#include "../directedGraph/Graph.h"

class BayesianNetwork {

    Graph<std::string, arma::mat> graph;

public:
    bool add(std::string);
    bool record(std::string, std::string, int, int, double);
    arma::mat get(std::string, std::map<std::string, int>);

};


#endif //GRAPH_BAYESIANNETWORK_H
