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

    auto* bayesNet = new BayesianNetwork();
    addFactors(bayesNet);

    REQUIRE(bayesNet->record("T", "E0", 0, 0, 0.33));

}

TEST_CASE("Connect factors with histogram", "[bayesNet]") {

    auto* bayesNet = new BayesianNetwork(3);
    addFactors(bayesNet);

    REQUIRE(bayesNet->record("T", "E0", 0, 0));
    REQUIRE(bayesNet->record("T", "E0", 1, 0));
    REQUIRE(bayesNet->record("T", "E0", 2, 0));

}

TEST_CASE("Erase", "[bayesNet") {

    auto* bayesNet = new BayesianNetwork(3);

    REQUIRE(!bayesNet->erase("T", "E0", 0, 0));

    addFactors(bayesNet);

    REQUIRE(!bayesNet->erase("T", "E0", 0, 0));

    REQUIRE(bayesNet->record("T", "E0", 0, 0));
    REQUIRE(bayesNet->record("T", "E0", 1, 0));
    REQUIRE(bayesNet->record("T", "E0", 2, 0));

    REQUIRE(bayesNet->erase("T", "E0", 0, 0));
    REQUIRE(bayesNet->erase("T", "E0", 1, 0));
    REQUIRE(bayesNet->erase("T", "E0", 2, 0));

    REQUIRE(!bayesNet->erase("T", "E0", 0, 0));
    REQUIRE(!bayesNet->erase("T", "E0", 1, 0));
    REQUIRE(!bayesNet->erase("T", "E0", 2, 0));

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

    const std::vector<double> THETA_HIDDEN = {0.25, 0.40, 0.35};
    const int SAMPLES = 10000;

    arma::rowvec dataHidden = bayesNet->simulateHiddenData(THETA_HIDDEN, SAMPLES);

    SECTION("Simulate hidden data") {

        REQUIRE(dataHidden.size() == SAMPLES);

        SECTION("Compute theta hidden") {

            arma::rowvec thetaHidden = bayesNet->computeThetaHidden(dataHidden);
            REQUIRE(thetaHidden.size() == bayesNet->getNumStates());

        }
    }

    SECTION("Simulate visible data") {

        addFactors(bayesNet);

        /*
         * E0:
         * 0.33 0.40 0.50
         * 0.33 0.25 0.20
         * 0.34 0.35 0.30
         */
        REQUIRE(bayesNet->record("T", "E0", 0, 0, 0.33));
        REQUIRE(bayesNet->record("T", "E0", 0, 1, 0.33));
        REQUIRE(bayesNet->record("T", "E0", 0, 2, 0.34));

        REQUIRE(bayesNet->record("T", "E0", 1, 0, 0.40));
        REQUIRE(bayesNet->record("T", "E0", 1, 1, 0.25));
        REQUIRE(bayesNet->record("T", "E0", 1, 2, 0.35));

        REQUIRE(bayesNet->record("T", "E0", 2, 0, 0.50));
        REQUIRE(bayesNet->record("T", "E0", 2, 1, 0.20));
        REQUIRE(bayesNet->record("T", "E0", 2, 2, 0.30));

        /*
         * E1:
         * 0.30 0.60 0.70
         * 0.65 0.20 0.10
         * 0.05 0.20 0.20
         */
        REQUIRE(bayesNet->record("T", "E1", 0, 0, 0.30));
        REQUIRE(bayesNet->record("T", "E1", 0, 1, 0.65));
        REQUIRE(bayesNet->record("T", "E1", 0, 2, 0.05));

        REQUIRE(bayesNet->record("T", "E1", 1, 0, 0.60));
        REQUIRE(bayesNet->record("T", "E1", 1, 1, 0.20));
        REQUIRE(bayesNet->record("T", "E1", 1, 2, 0.20));

        REQUIRE(bayesNet->record("T", "E1", 2, 0, 0.70));
        REQUIRE(bayesNet->record("T", "E1", 2, 1, 0.10));
        REQUIRE(bayesNet->record("T", "E1", 2, 2, 0.20));

        std::map<std::string, arma::rowvec> visibleData = bayesNet->simulateVisibleData("T", dataHidden, SAMPLES);

        SECTION("Compute theta visible") {
            std::map<std::string, arma::mat> thetaVisible = bayesNet->computeThetaVisible(dataHidden, visibleData);
        }
    }

    SECTION("Simulate visible data in and save it in data structure") {

        addFactors(bayesNet);

        arma::mat e0 = { {0.33, 0.40, 0.50},
                         {0.33, 0.25, 0.20},
                         {0.34, 0.35, 0.30} };

        arma::mat e1 = { {0.30, 0.60, 0.70},
                         {0.65, 0.20, 0.10},
                         {0.05, 0.20, 0.20} };

        std::map<std::string, arma::mat> thetaVisible = { {"E0", e0},
                                                          {"E1", e1} };

        std::map<std::string, arma::rowvec> dataVisible = bayesNet->simulateVisibleData(thetaVisible, "T", dataHidden, SAMPLES);

        SECTION("Compute theta visible") {
            std::map<std::string, arma::mat> calculatedThetaVisible = bayesNet->computeThetaVisible("T");
        }
    }
}

TEST_CASE("Actual runs", "[bayesNet") {

    /*
     * 1.  Simulate hidden data
     * 2.  Simulate visible data using hidden data
     * 3.  Generate random hidden data
     *
     * 4.  Compute theta hidden using random hidden data
     * 5.  Compute theta visible using random hidden data and actual visible data
     *
     * 6.  Impute hidden node using actual visible data, the random theta hidden and partly random theta visible.
     * 7.  Generate new hidden data using the results of the above step.
     *
     * 8.  Compute theta hidden using the new, slightly improved hidden data.
     * 9.  Compute theta visible using the new, slightly improved hidden data and actual visible data.
     *
     * 10. Repeat steps 6 - 9.
     */

    auto* bayesNet = new BayesianNetwork(3);
    addFactors(bayesNet);

    arma::mat e0 = { {0.33, 0.40, 0.50},
                     {0.33, 0.25, 0.20},
                     {0.34, 0.35, 0.30} };

    arma::mat e1 = { {0.30, 0.60, 0.70},
                     {0.65, 0.20, 0.10},
                     {0.05, 0.20, 0.20} };

    std::map<std::string, arma::mat> thetaVisible = { {"E0", e0},
                                                      {"E1", e1} };

    const std::vector<double> THETA_HIDDEN = {0.15, 0.50, 0.35};
    const int SAMPLES = 10000;

    arma::rowvec dataHidden = bayesNet->simulateHiddenData(THETA_HIDDEN, SAMPLES);
    std::map<std::string, arma::rowvec> dataVisible = bayesNet->simulateVisibleData(thetaVisible, "T", dataHidden, SAMPLES);

    arma::mat final = arma::mat(SAMPLES, THETA_HIDDEN.size());

    arma::rowvec randomDist = arma::rowvec(THETA_HIDDEN.size(), arma::fill::randu);
    dataHidden = bayesNet->simulateHiddenData(randomDist, dataHidden.size());

    arma::rowvec thetaHidden = bayesNet->computeThetaHidden(dataHidden);
    thetaVisible = bayesNet->computeThetaVisible(dataHidden, dataVisible);

    for (int j = 0; j < 800; ++j) {

        for (int i = 0; i < SAMPLES; ++i) {

            arma::mat tV = arma::mat(dataVisible.size(), THETA_HIDDEN.size());

            int count = 0;

            for (auto &&node : dataVisible) {
                tV.row(count++) = thetaVisible[node.first].row(node.second(i));
            }

            final.row(i) = bayesNet->imputeHiddenNode(thetaHidden, tV);

        }

        dataHidden = bayesNet->simulateHiddenData(arma::mean(final), final.n_rows);

        thetaHidden = bayesNet->computeThetaHidden(dataHidden);
        thetaVisible = bayesNet->computeThetaVisible(dataHidden, dataVisible);

    }

    std::cout << thetaHidden << std::endl;

}
