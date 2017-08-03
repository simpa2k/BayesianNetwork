//
// Created by simon on 2017-04-26.
//

#include "armadillo"
#include "BayesianNetwork.h"
#include <random>
#include <iostream>

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

/**
 * Utility method to generate data associated with a hidden node.
 *
 * @param thetaHidden A list of probabilities of the hidden node taking certain
 * values. Each probability describes the value indicated by its position in
 * the list. Thus, the list {0.38, 0.62} would indicate that there is a 38%
 * probability that the hidden node takes the value 0 and a 62% chance that it
 * takes the value 1.
 * @param samples The number of data points to generate.
 * @return A set of generated values.
 */
arma::rowvec BayesianNetwork::simulateHiddenData(const std::vector<double> thetaHidden, const int samples) {

    std::discrete_distribution<> dist(thetaHidden.begin(), thetaHidden.end()); // Create a custom distribution by providing an iterator to the list.
    std::mt19937 eng(std::time(0)); // Initiate a mersenne twister.

    arma::rowvec dataHidden(samples);

    for (int i = 0; i < samples; ++i) { // Generate the data points
        dataHidden(i) = dist(eng);
    }

    return dataHidden;

}

/**
 * Utility method to simulate visible data based. Requires data measured from a
 * hidden node to determine which probability distribution should be used to
 * generate the visible data.
 *
 * @param hiddenNode The key required to access the hidden node.
 * @param hiddenData The data of the hidden node.
 * @param samples The number of data points to generate.
 * @return A mapping of visible node keys to sets of generated data.
 */
std::map<std::string, arma::rowvec> BayesianNetwork::simulateVisibleData(const std::string hiddenNode,
                                                                         const arma::rowvec hiddenData,
                                                                         const int samples) {

    std::map<std::string, arma::mat> weights = graph.getWeights(hiddenNode); // Get all visible nodes that the hidden node is associated with.
    std::map<std::string, arma::rowvec> dataVisible;

    std::random_device rd;
    std::mt19937 eng(rd());

    for (auto &&node : weights) { // For each visible node.

        arma::rowvec simulatedDataPoints(samples);

        for (int i = 0; i < hiddenData.size(); ++i) { // For each hidden data point.

            double dataPoint = hiddenData(i);

            /*
             * The weight of each edge between the hidden and the
             * visible nodes is a matrix of probabilities. The value
             * of the hidden node is taken as a positional indicator
             * of which column of the matrix to look at.
             */
            arma::colvec col = node.second.col(dataPoint); // Pick out the column.

            std::discrete_distribution<> dist(col.begin(), col.end()); // Create a distribution from the set of probabilities contained in the column by providing an iterator.

            int simulatedDataPoint = dist(eng); // Generate the value.
            simulatedDataPoints(i) = simulatedDataPoint; // Put the generated visible value in the exact same position as the hidden value. This positional correspondence is what indicates causal correspondence.

        }

        dataVisible.insert(std::pair<std::string, arma::rowvec>(node.first, simulatedDataPoints)); // Insert the simulated data points under the key provided.

    }

    return dataVisible;

}

arma::uword BayesianNetwork::getNumStates() const {
    return numStates;
}

/**
 * Method to compute the probability of a hidden node taking certain values,
 * based on a set of data.
 *
 * @param dataHidden The measured values that the hidden node has taken.
 * @return The probability that the hidden node takes a certain value. The
 * position of each probability indicates which value it describes. 0.38
 * in position 0 would therefore indicate that there is a 38% probability
 * that the hidden node takes the value 0.
 */
arma::rowvec BayesianNetwork::computeThetaHidden(const arma::rowvec dataHidden) {

    arma::rowvec histogram(numStates, arma::fill::zeros);

    for (auto &&dataPoint : dataHidden) {
        ++histogram(dataPoint);
    }

    return histogram / dataHidden.size();

}

/**
 * Method to compute the probabilities of a series of visible nodes taking
 * certain values, given that a hidden node takes certain values, based on
 * lists of gathered data. Correspondence between hidden and visible data
 * is indicated by the values having the same position in the respective
 * lists of gathered data. Thus, if one measurement has been made and the
 * hidden node measured 1 when a visible node measured 2, the values would
 * both be placed in position 0.
 *
 * @param dataHidden A list of values that the hidden node has taken.
 * @param dataVisible A map of nodes with corresponding lists of values
 * that those nodes have taken.
 * @return The probabilities of the visible nodes taking certain values, given
 * that the hidden node has taken certain values.
 */
std::map<std::string, arma::mat>
BayesianNetwork::computeThetaVisible(arma::rowvec dataHidden, std::map<std::string, arma::rowvec> dataVisible) {

    std::map<std::string, arma::mat> histogramByNode;

    for (auto &&visibleFactor : dataVisible) { // For each visible node.

        arma::mat histogram(numStates, numStates, arma::fill::zeros); // Initialize a matrix to hold the counts of each measured value.

        /*
         * Create a histogram as a first step to calculate
         * the probabilities. The value of the hidden factor
         * determines which column to increment in, the visible
         * factor which row.
         */
        for (int i = 0; i < dataHidden.size(); ++i) {

            double hiddenDataPoint = dataHidden(i);
            double visibleDataPoint = visibleFactor.second(i);

            ++histogram(visibleDataPoint, hiddenDataPoint);

        }

        histogramByNode.insert(std::pair<std::string, arma::mat>(visibleFactor.first, histogram));

    }

    for (auto &&item : histogramByNode) {

        /*
         * Divide each column, element-wise, with the total number
         * of data points in the column to get the probability.
         */
        item.second.each_col([&dataHidden] (arma::vec& column) {
            column /= (float) arma::accu(column);
        });

        /*
         * Fix points where division by zero has occurred.
         */
        item.second.transform([] (double val) {
            return std::isnan(val) ? double(0) : val;
        });
    }

    return histogramByNode;

}

