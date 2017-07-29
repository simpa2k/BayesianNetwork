//
// Created by simon on 2017-04-26.
//

#include "catch.h"
#include "armadillo"
#include <iostream>
#include <random>
#include <ctime>

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
    REQUIRE(bayesNet->record("T", "E0", 1, 0, 0.40));

    REQUIRE(bayesNet->record("T", "E0", 0, 1, 0.33));
    REQUIRE(bayesNet->record("T", "E0", 1, 1, 0.25));

    REQUIRE(bayesNet->record("T", "E1", 0, 0, 0.25));
    REQUIRE(bayesNet->record("T", "E1", 1, 0, 0.60));

    REQUIRE(bayesNet->record("T", "E1", 0, 1, 0.75));
    REQUIRE(bayesNet->record("T", "E1", 1, 1, 0.95));

    std::string hidden = "T";
    std::map<std::string, arma::uword> query = { {"E0", 0},
                                                 {"E1", 1} };

    arma::mat correct = { {0.33, 0.40},
                          {0.75, 0.95} };

    arma::mat result = bayesNet->get(hidden, query);

    arma::umat evaluated = (result == correct);

    evaluated.for_each([] (arma::uword val) {
        REQUIRE(val == 1);
    });
}

TEST_CASE("Simulate data according to custom distributions", "[bayesNet") {

    BayesianNetwork* bayesNet = new BayesianNetwork(3);

    const std::vector<double> THETA_HIDDEN = {0.25, 0.60, 0.15};
    const int SAMPLES = 1000;

    //std::vector<int> dataHidden = bayesNet->simulateHiddenData(THETA_HIDDEN, SAMPLES);
    arma::rowvec dataHidden = bayesNet->simulateHiddenData(THETA_HIDDEN, SAMPLES);

    SECTION("Simulate hidden data") {

        REQUIRE(dataHidden.size() == SAMPLES);

        for (auto iter = dataHidden.begin(); iter != dataHidden.end(); ++iter) {
            //std::cout << (iter != dataHidden.begin() ? ", " : "") << *iter;
        }

        std::cout << std::endl;

        SECTION("Compute theta hidden") {

            arma::rowvec thetaHidden = bayesNet->computeThetaHidden(dataHidden);
            REQUIRE(thetaHidden.size() == bayesNet->getNumStates());

        }
    }

    SECTION("Simulate visible data") {

        addFactors(bayesNet);

        REQUIRE(bayesNet->record("T", "E0", 0, 0, 0.33));
        REQUIRE(bayesNet->record("T", "E0", 0, 1, 0.33));
        REQUIRE(bayesNet->record("T", "E0", 0, 2, 0.34));

        REQUIRE(bayesNet->record("T", "E0", 1, 0, 0.40));
        REQUIRE(bayesNet->record("T", "E0", 1, 1, 0.25));
        REQUIRE(bayesNet->record("T", "E0", 1, 2, 0.35));

        REQUIRE(bayesNet->record("T", "E1", 0, 0, 0.30));
        REQUIRE(bayesNet->record("T", "E1", 0, 1, 0.65));
        REQUIRE(bayesNet->record("T", "E1", 0, 2, 0.05));

        REQUIRE(bayesNet->record("T", "E1", 1, 0, 0.60));
        REQUIRE(bayesNet->record("T", "E1", 1, 1, 0.20));
        REQUIRE(bayesNet->record("T", "E1", 1, 2, 0.20));

        std::map<std::string, std::vector<int>> visibleData = bayesNet->simulateVisibleData("T", dataHidden, SAMPLES);

        SECTION("Compute theta visible") {
            arma::mat thetaVisible = bayesNet->computeThetaVisible(dataHidden, visibleData);
        }
    }
}

/*TEST_CASE("TEST") {

    arma::mat A = arma::mat(10, 10, arma::fill::zeros);
    A(9, 9) = 123.0;

    std::cout << A << std::endl;

}*/
