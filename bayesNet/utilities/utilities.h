/*
 * Miscellaneous utility functions that are used
 * both by the brain and the network. The computeThetaHidden and
 * computeThetaVisible functions are taken from Jace Kohlmeier's
 * Bayesian network example, and are just translations from his original
 * python code. The article can be found at http://derandomized.com/post/20009997725/bayes-net-example-with-python-and-khanacademy 
 * and all credit goes to him for the work on those. The expandVertically 
 * and expandHorizontally functions were written to mimic numpy's broadcast functionality
 * being used in Kohlmeier's original python code. They basically reshape the target matrix
 * but fill in the resulting new space with copies of the original content, making multiplication
 * with matrices with otherwise incompatible dimensions possible.
 */

#ifndef THETA_H
#define THETA_H

#include <armadillo>

using namespace arma;

double computeThetaHidden(umat*);
mat computeThetaVisible(umat*, umat*);
void expandVertically(mat*, int);
void expandHorizontally(umat*, int);

#endif
