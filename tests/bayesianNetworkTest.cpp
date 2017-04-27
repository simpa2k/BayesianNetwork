//
// Created by simon on 2017-04-26.
//

#include "catch.h"
#include "armadillo"

#include "../bayesNet/BayesianNetwork.h"

void addFactors(BayesianNetwork* bayesNet) {

    bayesNet->add("T");

    bayesNet->add("E0");
    bayesNet->add("E1");
    bayesNet->add("E2");
    bayesNet->add("E3");
    bayesNet->add("E4");

}

TEST_CASE("Add factor", "[bayesNet]") {

    BayesianNetwork bayesNet;
    REQUIRE(bayesNet.add("T"));

}

TEST_CASE("Connect factors", "[bayesNet]") {

    BayesianNetwork* bayesNet = new BayesianNetwork();
    addFactors(bayesNet);

    REQUIRE(bayesNet->record("T", "E0", 0, 0, 0.33));

}

TEST_CASE("Get probabilities", "[bayesNet]") {

    BayesianNetwork* bayesNet = new BayesianNetwork();
    addFactors(bayesNet);

    REQUIRE(bayesNet->record("T", "E0", 0, 0, 0.33));

    std::string hidden = "T";
    std::map<std::string, int> query = {{"E0", 0}};
    arma::rowvec correct = {0.25, 0.55};

    arma::mat result = bayesNet->get(hidden, query);
    //std::cout << "Got: " << bayesNet->get(hidden, query) << std::endl << "Correct: " << correct << std::endl;

}
