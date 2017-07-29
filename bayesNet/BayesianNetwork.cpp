//
// Created by simon on 2017-04-26.
//

#include "armadillo"
#include "BayesianNetwork.h"

BayesianNetwork::BayesianNetwork() = default;
BayesianNetwork::BayesianNetwork(arma::uword states) : numStates{states} {}

bool BayesianNetwork::add(std::string factorName) {
    return graph.add(factorName);
}

bool BayesianNetwork::record(std::string factor1, std::string factor2, arma::uword factor1State, arma::uword factor2State, double factor2Probability) {

    arma::mat values;
    arma::mat* probabilities = graph.getWeight(factor1, factor2);

    if (probabilities != NULL) {
        values = *probabilities;
    } else {
        values = arma::mat(numStates, numStates, arma::fill::zeros);
    }

    values(factor2State, factor1State) = factor2Probability;

    bool result = graph.connect(factor1, factor2, values);
    delete probabilities;

    return result;

}

arma::mat BayesianNetwork::get(std::string hidden, std::map<std::string, arma::uword> visibleStates) {

    arma::mat currentStates;

    for (auto const& it : visibleStates) {

        arma::mat* probabilities = graph.getWeight(hidden, it.first);
        currentStates = arma::join_cols(currentStates, probabilities->row(it.second));

        delete probabilities;

    }

    return currentStates;

}

std::vector<int> BayesianNetwork::simulateHiddenData(const std::vector<double> thetaHidden, const int samples) {

    std::discrete_distribution<> dist(thetaHidden.begin(), thetaHidden.end());
    std::mt19937 eng(std::time(0));

    std::vector<int> dataHidden;

    for (int i = 0; i < samples; ++i) {
        dataHidden.push_back(dist(eng));
    }

    return dataHidden;

}

std::map<std::string, std::vector<int>> BayesianNetwork::simulateVisibleData(const std::string hiddenNode, const std::vector<int> hiddenData, const int samples) {

    std::map<std::string, arma::mat> weights = graph.getWeights(hiddenNode);
    std::map<std::string, std::vector<int>> dataVisible;

    for (auto &&dataPoint : hiddenData) {

        for (auto &&node : weights) {

            arma::colvec col = node.second.col(dataPoint);

            std::mt19937 eng(std::time(0));

            std::discrete_distribution<> dist(col.begin(), col.end());
            std::vector<int> simulatedDataPoints;

            for (int i = 0; i < samples; ++i) {

                int simulatedDataPoint = dist(eng);
                simulatedDataPoints.push_back(simulatedDataPoint);

            }

            dataVisible.insert(std::pair<std::string, std::vector<int>>(node.first, simulatedDataPoints));

        }
    }

    return dataVisible;
}

