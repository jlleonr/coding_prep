from typing import List
from collections import deque

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