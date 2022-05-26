from typing import List, Tuple
from collections import deque
import sys

sys.path.append("../coding_prep/")

import variables as vars


graph1 = {
    "a": ["b", "c"],
    "b": ["d"],
    "c": ["e"],
    "d": [],
    "e": [],
    "f": ["d"]
}

flights = {
    "DFW": ["SC", "ATL"],
    "SC": ["MIA"],
    "MIA": ["SJU"],
    "SJU": [],
    "ATL": ["SJU"],
}

graph2 = {
    "w":["x", "v"],
    "x":["w", "y"],
    "y":["x", "z"],
    "z":["y", "v"],
    "v":["w", "z"]
}

'''
A Graph class that takes a list of tuples
and converts the tuples to the appropriate
nodes and edges
'''
class Graph:

    def __init__(self, edges: tuple):
        self.graph = edges

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, edges):
        self._graph = {}

        for start, end in edges:
            # if start in self._graph:
            #     self._graph[start].append(end)
            # else:
            #     if end == "":
            #         self._graph[start] = []
            #     else:
            #         self._graph[start] = [end]
            if start not in self._graph and start != "":
                self._graph[start] = []
            if end not in self._graph and end != "":
                self._graph[end] = []
            if start != "":
                self._graph[start].append(end)
            if end != "":
                self._graph[end].append(start)


routes_graph = Graph(vars.flights)
print(routes_graph.graph)

'''
DFT isPathDFTRecursive approach
Returns if there is a path from source to destination
also returns the path if it exists
'''
def isPathDFTRecursive(graph: dict, source: str, destination: str, path: List[str]= [], visited: set = set()) -> List[str]:

    # Append current node to the path list
    path.append(source)

    # If we reached the destination node
    # return True
    if source == destination:
        return True

    # Else do a DFT for all source node's neighbors
    for neighbor in graph[source]:

        # Check that current node has not been visited
        if neighbor not in visited:

            # Add neighbor to visited set ** set() reads are O(1) **
            visited.add(neighbor)
            if isPathDFTRecursive(graph, neighbor, destination, path):
                return True

    path.pop()

    return False

path = []
ip = isPathDFTRecursive(graph1, "a", "e", path)
print(f"isPathDFTRecursive: {ip},{path}")

'''DFT iterative approach'''
def depthFirstTraversalIterative(graph: dict, source: str, path: List[str] = None):
    if path == None:
        path = []
    '''Create a stack to store nodes that form the DFT path'''
    stack = []
    stack.append(source)

    while len(stack) > 0:
        '''Pop the stack and append the popped node to the path list'''
        current = stack.pop()
        path.append(current)

        '''Push all neighbors from current node to the stack'''
        for neighbor in graph[current]:
            stack.append(neighbor)

    return path

'''
Find if there is a path in an undirected non-cyclic graph
from source node to destination node
'''
def hasPathDFTRecursive(graph: dict, src: str, dst: str) -> bool:
    if src == dst:
        return True

    for neighbor in graph[src]:
        if hasPathDFTRecursive(graph, neighbor, dst):
            return True

    return False

'''
BFT
'''
def shortestPathBFT(graph: dict, src: str, dst: str, path: List[str] = [], visited: set = set()) -> bool:
    # Add src to queue

    # while queue is not empty
    # pop the queue
    # check if src == dst
    # for every neighbor in node
    # add node to queue

    # if we reach the end return, false

    queue: deque = deque()
    queue.appendleft(src)

    while len(queue) > 0:

        currentNode: str = queue.popleft()

        visited.add(currentNode)
        path.append(currentNode)

        if currentNode == dst:
            return True

        for neighbor in graph[currentNode]:
            if neighbor not in visited:
                queue.appendleft(neighbor)

    return False


'''
hastPath function using BFT
'''
def hasPathBFT(graph: dict, src: str, dest: str) -> bool:
    queue: List[str] = [src]

    while(len(queue) > 0):
        current: str = queue.pop(0)

        if current == dest:
            return True

        for neighbor in graph[current]:
            queue.append(neighbor)

    return False


def shortestPath(graph: dict, src: str, dst: str) -> int:

    print(graph)

    queue: deque = deque()
    queue.append((src, 0))
    node: str
    distance: int
    visited: set = set()

    #implement BFT algorithm
    while len(queue) > 0:

        #get a node, distance from the queue
        (node, distance) = queue.popleft()

        #if the node from the queue == dst
        #return the distance
        if node == dst:
            return distance

        #Find the distance for every neighbor of the source
        for neighbor in graph[node]:
            #If neighbor hasn't been visited
            #add it to the set
            if neighbor not in visited:
                visited.add(neighbor)
                #add visited neighbor to the queue and increase the distance
                queue.append((neighbor, distance + 1))

    return -1

# *** Keep working to fix it
def shortestPath2(graph: dict, src: str, dst: str) -> Tuple[int, List[str]]:

    print(graph)

    path: List[str] = []
    current_path: List[str] = [src]
    queue: deque = deque()
    visited: set = set()
    node: str
    distance: int

    queue.append((src, 0, path))

    #implement BFT algorithm
    while len(queue) > 0:

        #get a node, distance from the queue
        (node, distance, current_path) = queue.popleft()

        #if the node from the queue == dst
        #return the distance
        if node == dst:
            return (distance, current_path)

        #Find the distance for every neighbor of the source
        for neighbor in graph[node]:

            current_path = list(path)
            current_path.append(neighbor)
            #queue.append(current_path)

            #If neighbor hasn't been visited
            #add it to the set
            if neighbor not in visited:
                visited.add(neighbor)
                #add visited neighbor to the queue and increase the distance
                queue.append((neighbor, distance + 1, current_path))

    return (-1, [])

# dft2 = depthFirstTraversalIterative(graph1, "a")
# print(dft2)

# print("\n")

# hpDFT = hasPathDFTRecursive(graph=graph1, src="a", dst="e")
# print(f"Is there a path from a --> e? ", hpDFT)

# hpDFT2 = hasPathDFTRecursive(graph=graph1, src="c", dst="f")
# print(f"Is there a path from c --> f? ", hpDFT2)

# assert hpDFT2 == False, "Error!"
path = []
src = "f"
dst = "d"
sp = shortestPathBFT(graph=graph1, src=src, dst=dst, path=path)
print(f"What is the shortest path from {src} to {dst}?: -> {path}")

src = "Mumbai"
dst = "New York"

sp2 = shortestPath2(graph=vars.flights_graph, src=src, dst=dst)
print(f"The shortest path from {src} to {dst} = ", sp2)