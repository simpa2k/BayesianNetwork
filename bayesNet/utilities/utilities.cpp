#ifndef THETA
#define THETA

#include <armadillo>
#include "utilities.h"

using namespace arma;

double computeThetaHidden(umat* dataHidden) {
    return mean(mean(conv_to<mat>::from(*dataHidden)));
}

mat computeThetaVisible(umat* dataHidden, umat* dataVisible) {

    mat thetaVisible = mat(dataVisible->n_cols, 2, fill::zeros);

    for (int i = 0; i < dataVisible->n_cols; ++i) {
        
        umat visibleCol = trans(dataVisible->col(i));

        thetaVisible.at(i, 0) = accu(visibleCol % (1 - *dataHidden)) / (float) accu(1 - *dataHidden);
        thetaVisible.at(i, 1) = accu(visibleCol % *dataHidden) / (float) accu(*dataHidden);

    }
    thetaVisible.transform( [] (double val) { return (std::isnan(val) ? double(0) : val); });

    return thetaVisible;
}

void expandVertically(mat* target, int targetRows) {

    target->resize(targetRows, target->n_cols);

    for (int i = 1; i < target->n_rows; ++i) {
        target->row(i) = target->row(0);
    }

}

void expandHorizontally(umat* target, int targetColumns) {

    target->resize(target->n_rows, targetColumns);

    for (int i = 0; i < target->n_rows; ++i) {
        
        double value = target->at(i, 0);
        target->row(i).fill(value);

    }
}

#endif
