//
// Created by simon on 2017-04-03.
//

#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <map>
#include <vector>
#include <queue>
#include <algorithm>

template <typename T, typename W>
struct edge;

template <typename T, typename W>
struct node;

template <typename T, typename W>
struct edge {

    node<T, W>* target;
    W weight;

};

template <typename T, typename W>
struct node {

    T data;
    int indegree = 0;

    std::vector<edge<T, W>> edges;

};

template <class T, class W>
class Graph {

    std::map<T, node<T, W>> nodes;
    bool areConnected(node<T, W> first, node<T, W> second);

public:

    bool add(T data);
    bool connect(T node1, T node2, W weight);
    std::vector<T> topologicalSort();

};

template <typename T, typename W>
bool Graph<T, W>::add(T data) {

    typename std::map<T, node<T, W>>::iterator existing = nodes.find(data);

    if (existing != nodes.end()) {
        return false;
    }

    node<T, W>* newNode = new node<T, W>;
    newNode->data = data;

    nodes[data] = *newNode;

    return true;

}

template <typename T, typename W>
bool Graph<T, W>::connect(T node1, T node2, W weight) {

    typename std::map<T, node<T, W>>::iterator existing1 = nodes.find(node1);
    typename std::map<T, node<T, W>>::iterator existing2 = nodes.find(node2);

    if (existing1 == nodes.end() || existing2 == nodes.end() || areConnected(existing1->second, existing2->second)) {
        return false;
    }

    edge<T, W>* connection = new edge<T, W>;
    connection->target = &existing2->second;
    connection->weight = weight;

    existing1->second.edges.push_back(*connection);
    existing2->second.indegree++;

    return true;
}

template <typename T, typename W>
std::vector<T> Graph<T, W>::topologicalSort() {

    std::vector<T> topologicalOrdering;
    std::map<T, int> indegrees;
    std::queue<node<T, W>> queue;

    for (auto const& it : nodes) {

        if (it.second.indegree == 0) {
            queue.push(it.second);
        } else {
            indegrees[it.second.data] = it.second.indegree;
        }
    }

    while (!queue.empty()) {

        node<T, W> n = queue.front();
        queue.pop();

        topologicalOrdering.push_back(n.data);

        for (auto const& it : n.edges) {

            int* indegree = &indegrees[it.target->data];

            if (--*indegree == 0) {
                queue.push(*it.target);
            }
        }
    }
    return topologicalOrdering;

}

template <typename T, typename W>
bool Graph<T, W>::areConnected(node<T, W> first, node<T, W> second) {

    std::vector<edge<T, W>> &edges = first.edges;
    auto it = std::find_if(edges.begin(), edges.end(), [&second](const edge<T, W>& edge) {
        return edge.target->data == second.data;
    });

    return it != edges.end();

}

#endif //GRAPH_GRAPH_H
