#include <armadillo>
#include "Brain.h"
#include "../utilities/utilities.h"

using namespace std;
using namespace arma;

Brain::Brain(int learningIterations) : learningIterations(learningIterations) {};

void Brain::imputeHiddenNode(arma::umat * dataHidden, arma::umat * dataVisible, double thetaHidden, arma::mat thetaVisible) {

    arma::mat colZero = arma::trans(thetaVisible.col(0));
    arma::mat colOne = arma::trans(thetaVisible.col(1));

    expandVertically(&colZero, dataVisible->n_rows);
    expandVertically(&colOne, dataVisible->n_rows);

    arma::mat probVis0 = colZero % *dataVisible + (1 - colZero) % (1 - *dataVisible);
    arma::mat probVis0Unnorm = (1 - thetaHidden) * prod(probVis0, 1);

    arma::mat probVis1 = colOne % *dataVisible + (1 - colOne) % (1 - *dataVisible);
    arma::mat probVis1Unnorm = thetaHidden * prod(probVis1, 1);

    arma::mat hidden = probVis1Unnorm / (probVis0Unnorm + probVis1Unnorm);
    hidden.transform( [] (double val) { return (std::isnan(val) ? double(0) : val); });

    *dataHidden = arma::trans(hidden > arma::mat(hidden.n_rows, hidden.n_cols, arma::fill::randu));

}

double Brain::learn(arma::umat dataHidden, arma::umat dataVisible) {
    
    double thetaHidden = computeThetaHidden(&dataHidden);
    mat thetaVisible = computeThetaVisible(&dataHidden, &dataVisible);

    for (int i = 0; i < learningIterations; ++i) {
       
       imputeHiddenNode(&dataHidden, &dataVisible, thetaHidden, thetaVisible); 

       /*if (computeThetaHidden(&dataHidden) < 0.5) {
           dataHidden = 1 - dataHidden;
       }*/

       thetaHidden = computeThetaHidden(&dataHidden);
       thetaVisible = computeThetaVisible(&dataHidden, &dataVisible);

    }
    return thetaHidden;
}
