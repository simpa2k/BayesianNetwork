//
// Created by simon on 2017-04-26.
//

#include "catch.h"
#include "armadillo"

#include "../bayesNet/BayesianNetwork.h"

void addFactors(BayesianNetwork bayesNet) {

    bayesNet.add("T");

    bayesNet.add("E0");
    bayesNet.add("E1");
    bayesNet.add("E2");
    bayesNet.add("E3");
    bayesNet.add("E4");

}

TEST_CASE("Add factor", "[bayesNet]") {

    BayesianNetwork bayesNet;
    REQUIRE(bayesNet.add("T"));

}

TEST_CASE("Connect factors", "[bayesNet]") {

    BayesianNetwork bayesNet;
    addFactors(bayesNet);

    REQUIRE(bayesNet.connect("T", "E0", 0, 0.25, {0.55}));

}

TEST_CASE("Get probabilities", "[bayesNet]") {

    BayesianNetwork bayesNet;
    addFactors(bayesNet);

    std::map<std::string, int> query = {{"T", 0}};
    arma::rowvec correct = {0.25, 0.55};

    std::cout << "Got: " << bayesNet.get(query) << std::endl << "Correct: " << correct << std::endl;

}
