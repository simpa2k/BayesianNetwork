#include <iostream>
#include "armadillo"
#include "directedGraph/Graph.h"
#include "bayesNet/BayesianNetwork.h"

/*void addFactors(BayesianNetwork* bayesNet) {

    bayesNet->add("T");

    bayesNet->add("E0");
    bayesNet->add("E1");
    bayesNet->add("E2");
    bayesNet->add("E3");
    bayesNet->add("E4");

}

int main() {

    BayesianNetwork* bayesNet = new BayesianNetwork();
    addFactors(bayesNet);

    bayesNet->record("T", "E0", 0, 0, 0.33);

    std::string hidden = "T";
    std::map<std::string, arma::uword> query = {{"E0", 0}};
    arma::rowvec correct = {0.25, 0.55};

    arma::mat result = bayesNet->get(hidden, query);

    return 0;
}*/