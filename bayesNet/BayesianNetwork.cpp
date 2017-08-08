//
// Created by simon on 2017-04-26.
//

#include "armadillo"
#include "BayesianNetwork.h"
#include "utilities/utilities.h"
#include <random>
#include <iostream>

BayesianNetwork::BayesianNetwork() = default;
BayesianNetwork::BayesianNetwork(arma::uword states) : numStates{states} {}

/**
 * Method for adding a node to the network.
 *
 * @param factorName A descriptive name for the factor.
 * @return A boolean value indicating whether or not the
 * factor was added. False will be returned if there already
 * is a factor with the provided name in the network.
 */
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

bool BayesianNetwork::record(std::string factor1, std::string factor2, arma::uword factor1State, arma::uword factor2State) {

    arma::mat values;
    arma::mat* probabilities = graph.getWeight(factor1, factor2);

    if (probabilities != NULL) {
        values = *probabilities;
    } else {
        values = arma::mat(numStates, numStates, arma::fill::zeros);
    }

    ++values(factor2State, factor1State);

    bool result = graph.connect(factor1, factor2, values);
    delete probabilities;

    return result;

}


bool BayesianNetwork::erase(std::string factor1, std::string factor2, arma::uword factor1State, arma::uword factor2State) {

    arma::mat values;
    arma::mat* probabilities = graph.getWeight(factor1, factor2);

    if (probabilities == NULL) {
        return false;
    }

    values = *probabilities;

    if (values(factor2State, factor1State) == 0) {
        return false;
    }

    --values(factor2State, factor1State);

    bool result = graph.connect(factor1, factor2, values);
    delete probabilities;

    return result;

}

/**
 * Method for getting the probabilities for all the states of a hidden node,
 * given that a series of visible nodes take certain values. The thought is
 * that this will be used when there has just been measurements of visible
 * nodes and the relevant hidden probabilities, given those measurements,
 * have to be established.
 *
 * @param hidden The name of the hidden node.
 * @param visibleStates A mapping of visible node keys to their states. This
 * will most likely be recently measured states.
 * @return A matrix of probabilities where each row represents a visible node.
 */
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

arma::rowvec BayesianNetwork::simulateHiddenData(arma::rowvec thetaHidden, int samples) {

    std::discrete_distribution<> dist(thetaHidden.begin(), thetaHidden.end()); // Create a custom distribution by providing an iterator to the list.
    std::mt19937 eng(std::time(0)); // Initiate a mersenne twister.

    arma::rowvec dataHidden(samples);

    for (int i = 0; i < samples; ++i) { // Generate the data points
        dataHidden(i) = dist(eng);
    }

    return dataHidden;

}

/**
 * Utility method to simulate visible data based on a set of hidden data.
 * Requires data measured from a hidden node to determine which probability
 * distribution should be used to generate the visible data.
 *
 * @param hiddenNode The key required to access the hidden node.
 * @param hiddenData The data of the hidden node.
 * @param samples The number of data points to generate.
 * @return A mapping of visible node keys to lists of generated data.
 */
std::map<std::string, arma::rowvec> BayesianNetwork::simulateVisibleData(const std::string hiddenNode,
                                                                         const arma::rowvec hiddenData,
                                                                         const int samples) {

    std::map<std::string, arma::mat> weights = graph.getWeights(hiddenNode); // Get all visible nodes that the hidden node is associated with, and their weights.
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
            simulatedDataPoints(i) = simulatedDataPoint; // Put the generated visible value in the exact same position as the hidden value. This positional correspondence is what indicates causal or temporal correspondence.

        }

        dataVisible.insert(std::pair<std::string, arma::rowvec>(node.first, simulatedDataPoints)); // Insert the simulated data points under the key provided.

    }

    return dataVisible;

}

std::map<std::string, arma::rowvec> BayesianNetwork::simulateVisibleData(std::map<std::string, arma::mat> thetaVisible,
                                                                          std::string hiddenNode,
                                                                          arma::rowvec hiddenData,
                                                                          int samples) {

    std::map<std::string, arma::rowvec> dataVisible;

    std::random_device rd;
    std::mt19937 eng(rd());

    for (auto &&node : thetaVisible) { // For each visible node.

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
            simulatedDataPoints(i) = simulatedDataPoint; // Put the generated visible value in the exact same position as the hidden value. This positional correspondence is what indicates causal or temporal correspondence.

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

std::map<std::string, arma::mat> BayesianNetwork::computeThetaVisible(std::string hiddenNode) {

    std::map<std::string, arma::mat> histogramByNode = graph.getWeights(hiddenNode);

    for (auto &&item : histogramByNode) {

        /*
         * Divide each column, element-wise, with the total number
         * of data points in the column to get the probability.
         */
        item.second.each_col([] (arma::vec& column) {
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

arma::rowvec BayesianNetwork::imputeHiddenNode(arma::rowvec thetaHidden, arma::mat thetaVisible) {

    arma::rowvec final = arma::rowvec(thetaHidden.size());

    for (arma::uword i = 0; i < thetaVisible.n_cols; ++i) {

        arma::mat copy = thetaVisible;
        copy.shed_col(i);

        arma::mat toBeSummed = arma::mat(1, thetaHidden.size());

        /*
         * Use everything that is not the current hidden probability
         * as a collective "not true" value but be sure to use the
         * theta hidden associated with each non-true column.
         */
        for (arma::uword j = 0; j < copy.n_cols; ++j) {

            arma::mat column = arma::trans(copy.col(j));

            double correctThetaHidden = thetaHidden(((i + 1) + j) % thetaHidden.size());
            arma::mat probVis0Unnorm = correctThetaHidden * arma::prod(column, 1); // i will never be greater than the number of hidden states, since the columns in thetaVisible in fact represent those states.

            toBeSummed.col(j) = probVis0Unnorm;
            
        }

        /*
         * Use the current hidden probability as the sole "true value".
         * Then repeat the process so that every hidden probability
         * has been treated as the true value.
         */
        arma::mat column = arma::trans(thetaVisible.col(i));

        arma::mat probVis1Unnorm = thetaHidden(i) * arma::prod(column, 1); // i will never be greater than the number of hidden states, since the columns in thetaVisible in fact represent those states.
        toBeSummed.tail_cols(1) = probVis1Unnorm;

        arma::mat hidden = probVis1Unnorm / sum(toBeSummed, 1); // sum rows

        final.col(i) = hidden;


    }

    return final;

}



