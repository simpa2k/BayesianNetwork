//
// Created by simon on 2017-04-26.
//

#include "armadillo"
#include "BayesianNetwork.h"

bool BayesianNetwork::add(std::string) {
    return true;
}

bool BayesianNetwork::connect(std::string, std::string, int, double, arma::rowvec) {
    return true;
}

arma::rowvec BayesianNetwork::get(std::map<std::string, int>) {
    return arma::rowvec();
}

