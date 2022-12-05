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
    "w": ["x", "v"],
    "x":  ["w", "y"],
    "y": ["x", "z"],
    "z": ["y", "v"],
    "v": ["w", "z"]
}


class Graph:
    '''
    A Graph class that takes a list of tuples
    and converts the tuples to the appropriate
    nodes and edges
    '''
    def __init__(self, edges: tuple):
        self.graph = edges

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, edges):
        self._graph = {}

        for start, end in edges:
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


def is_path_dft_recursive(graph: dict,
                          source: str,
                          destination: str,
                          path: List[str] = [],
                          visited: set = set()) -> List[str]:
    '''
    DFT isPathDFTRecursive approach
    Returns if there is a path from source to destination
    also returns the path if it exists
    '''
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
            if is_path_dft_recursive(graph, neighbor, destination, path):
                return True

    path.pop()

    return False


path = []
ip = is_path_dft_recursive(graph1, "a", "e", path)
print(f"is_path_dft_recursive: {ip},{path}")


def depth_first_traversal_iterative(graph: dict,
                                    source: str,
                                    path: List[str] = None) -> List[str]:
    '''DFT iterative approach'''
    if path is None:
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


def breadth_first_traversal_queue(graph: dict, source: str) -> List[str]:
    '''
    Breadth First Traversal implemented with a Queue,
    Queue has T(n) = O(1)
    '''
    queue: deque = deque()
    queue.append(source)
    path: List[str] = []

    while(len(queue) > 0):
        current = queue.popleft()
        path.append(current)
        for neighbor in graph[current]:
            queue.append(neighbor)

    return path


def has_path_dft_recursive(graph: dict, src: str, dst: str) -> bool:
    '''
    Find if there is a path in an undirected non-cyclic graph
    from source node to destination node
    '''
    if src == dst:
        return True
    for neighbor in graph[src]:
        if has_path_dft_recursive(graph, neighbor, dst):
            return True
    return False


def shortest_path_bft(graph: dict,
                      src: str,
                      dst: str,
                      path: List[str] = [],
                      visited: set = set()) -> bool:
    '''
    BFT
    '''
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

        current_node: str = queue.popleft()

        visited.add(current_node)
        path.append(current_node)

        if current_node == dst:
            return True

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.appendleft(neighbor)

    return False


def has_path_bft(graph: dict, src: str, dest: str) -> bool:
    '''
    hastPath function using BFT
    '''
    queue: List[str] = [src]

    while(len(queue) > 0):
        current: str = queue.pop(0)

        if current == dest:
            return True

        for neighbor in graph[current]:
            queue.append(neighbor)

    return False


def build_graph(edges: List[List[str]]) -> dict:
    '''
    Helper function to build a graph from a given list
    of edges
    '''
    graph: dict = {}

    for a, b in edges:

        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []

        graph[a].append(b)
        graph[b].append(a)

    return graph


class ConnectedComponents:
    '''
    For a given graph, count the number of connected components
    '''
    # Set the input graph value to the graph property
    def __init__(self, graph: dict) -> None:
        self.graph = graph

    # Property to read the graph variable as a property
    @property
    def graph(self) -> dict:
        return self.__graph

    # Setter for the graph property
    @graph.setter
    def graph(self, graph: dict) -> None:
        self.__graph = graph

    def count(self) -> int:
        '''
        Count how many connected groups are present in the
        input graph
        '''
        visited: set = set()
        count: int = 0

        # Do a Depth First Search for every node in the graph
        # After exploring all neighbors, increase counter for
        for node in self.graph:
            if self.__explored_all_neighbors(node, visited):
                count += 1
        return count

    def __explored_all_neighbors(self, current: str, visited: set) -> bool:
        if current in visited:
            return False

        visited.add(current)

        for neighbor in self.graph[current]:
            self.__explored_all_neighbors(neighbor, visited)

        return True


cc = ConnectedComponents(vars.graph2)
num_of_cc = cc.count()

print(f"The number of connected components = {num_of_cc}")


def shortest_path(graph: dict, src: str, dst: str) -> int:
    print(graph)

    queue: deque = deque()
    queue.append((src, 0))
    node: str
    distance: int
    visited: set = set()

    # implement BFT algorithm
    while len(queue) > 0:

        # get a node, distance from the queue
        (node, distance) = queue.popleft()

        # if the node from the queue == dst
        # return the distance
        if node == dst:
            return distance

        # Find the distance for every neighbor of the source
        for neighbor in graph[node]:
            # If neighbor hasn't been visited
            # add it to the set
            if neighbor not in visited:
                visited.add(neighbor)
                # add visited neighbor to the queue and increase the distance
                queue.append((neighbor, distance + 1))

    return -1


# *** Keep working to fix it
def shortest_path_2(graph: dict, src: str, dst: str) -> Tuple[int, List[str]]:

    print(graph)

    path: List[str] = []
    current_path: List[str] = [src]
    queue: deque = deque()
    visited: set = set()
    node: str
    distance: int

    queue.append((src, 0, path))

    # implement BFT algorithm
    while len(queue) > 0:
        # get a node, distance from the queue
        (node, distance, current_path) = queue.popleft()
        # if the node from the queue == dst
        # return the distance
        if node == dst:
            return (distance, current_path)

        # Find the distance for every neighbor of the source
        for neighbor in graph[node]:

            current_path = list(path)
            current_path.append(neighbor)
            # queue.append(current_path)

            # If neighbor hasn't been visited
            # add it to the set
            if neighbor not in visited:
                visited.add(neighbor)
                # add visited neighbor to the queue and increase the distance
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
sp = shortest_path_bft(graph=graph1, src=src, dst=dst, path=path)
print(f"What is the shortest path from {src} to {dst}?: -> {path}")

src = "Mumbai"
dst = "New York"

sp2 = shortest_path_2(graph=vars.flights_graph, src=src, dst=dst)
print(f"The shortest path from {src} to {dst} = ", sp2)


def count_islands(graph: dict) -> int:
    '''
    explore all neighbors of a node
    after all neighbors are visited, increase counter
    '''
    counter: int = 0
    visited: set = set()
    # For very node in the graph, explore all edges
    for node in graph:
        if explored_all_edges(graph, node, visited):
            counter += 1
    return counter


def explored_all_edges(graph: dict, node: str, visited: set) -> bool:

    # If a node has been visited, return false
    if node in visited:
        return False

    # Add visited node to the set
    visited.add(node)

    # explore all neighbors of the current node
    for neighbor in graph[node]:
        explored_all_edges(graph, neighbor, visited)

    return True


num_of_islands = count_islands(vars.graph2)
print(f"Number of islands = {num_of_islands}")


def find_largest_island(graph: dict) -> int:
    visited: set = set()
    biggest: int = 0
    # For every node in the graph, explore all edges
    for node in graph:
        size = explore_island_size(graph, node, visited)
        biggest = max(biggest, size)
    return biggest


def explore_island_size(graph: dict, node: str, visited: set) -> int:
    # If a node has been visited, return false
    if node in visited:
        return 0

    visited.add(node)
    # Initial island size
    size: int = 1

    # explore all nodes of the current island
    for neighbor in graph[node]:
        size += explore_island_size(graph, neighbor, visited)
    return size


biggest_island_size: int = find_largest_island(vars.graph2)
print(f"The size of the biggest Island in the graph = {biggest_island_size}")


def island_count(grid: List[List[str]]) -> int:
    '''
    Count the number of islands on a grid
    '''
    visited: set = set()
    count: int = 0

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if explore_dft(grid, row, col, visited) is True:
                count += 1
    return count


def explore_dft(grid: List[List[str]],
                row: int,
                col: int,
                visited: set) -> bool:
    '''
    Explore connected islands on a grid using DFT
    '''

    row_inbound: bool = (0 <= row) and (row < len(grid))
    col_inbound: bool = (0 <= col) and (col < len(grid[0]))

    if (not row_inbound) or (not col_inbound):
        return False

    # Check curremt coordinate is not Water
    if grid[row][col] == "W":
        return False

    # Store current position in the grid
    position: tuple = (row, col)

    # Check that position has not been visited
    if position in visited:
        return False

    visited.add(position)

    # Explore neighbors of current position in all cardinal points
    explore_dft(grid, row - 1, col, visited)  # Explore up
    explore_dft(grid, row + 1, col, visited)  # Explore down
    explore_dft(grid, row, col - 1, visited)  # Explore left
    explore_dft(grid, row, col + 1, visited)  # Explore right

    # If all non water neighbors have been explored and
    # coordinate is not Water, then we found an island
    return True


def minimumIsland(grid: List[List[str]]) -> int:
    visited: set = set()
    min_size: int = len(grid) * len(grid[0])

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            size = explore_minimum_dft(grid, row, col, visited)
            if size > 0:
                min_size = min(size, min_size)

    return min_size


def explore_minimum_dft(grid: List[List[str]],
                        row: int,
                        col: int,
                        visited: set) -> int:

    rowInbound: bool = (0 <= row) and (row < len(grid))
    colInbound: bool = (0 <= col) and (col < len(grid[0]))

    if (not rowInbound) or (not colInbound):
        return 0

    # Check current coordinate is not Water
    if grid[row][col] == "W":
        return 0

    # Store current position in the grid
    position: tuple = (row, col)

    # Check that position has not been visited
    if position in visited:
        return 0

    visited.add(position)

    size = 1

    # Explore neighbors of current position in all cardinal points
    size += explore_minimum_dft(grid, row - 1, col, visited)  # Explore up
    size += explore_minimum_dft(grid, row + 1, col, visited)  # Explore down
    size += explore_minimum_dft(grid, row, col - 1, visited)  # Explore left
    size += explore_minimum_dft(grid, row, col + 1, visited)  # Explore right

    # If all non water neighbors have been explored and
    # coordinate is not Water, then we found an island
    return size
