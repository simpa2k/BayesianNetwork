#include <iostream>
#include "directedGraph/Graph.h"

/*int main() {

    Graph<std::string, double> graph;

    graph.add("1");
    graph.add("2");
    graph.add("3");
    graph.add("4");
    graph.add("5");
    graph.add("6");
    graph.add("7");

    graph.connect("1", "2", 1.0);
    graph.connect("1", "3", 1.0);

    graph.connect("2", "4", 1.0);
    graph.connect("2", "5", 1.0);

    graph.connect("3", "6", 1.0);

    graph.connect("4", "3", 1.0);
    graph.connect("4", "6", 1.0);
    graph.connect("4", "7", 1.0);

    graph.connect("5", "4", 1.0);
    graph.connect("5", "7", 1.0);

    graph.connect("7", "6", 1.0);

    std::vector<std::string> topologicalOrdering = graph.topologicalSort();

    for (auto const& it : topologicalOrdering) {
        std::cout << it << std::endl;
    }

    return 0;
}*/