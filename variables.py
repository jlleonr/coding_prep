from typing import List


flights: tuple = [
    ("Mumbai", "Paris"),
    ("Mumbai", "Dubai"),
    ("Paris", "Dubai"),
    ("Paris", "New York"),
    ("Dubai", "New York"),
    ("New York", "Toronto"),
    ("Toronto", "")
]

'''
Source Edges
'''
edges: List[List[str]] = [
    ["i", "j"],
    ["k", "i"],
    ["m", "k"],
    ["k", "l"],
    ["o", "n"]
]

'''
Source graph
'''
graph = {
    "a": ["c", "b"],
    "b": ["d"],
    "c": ["e"],
    "d": ["f"],
    "e": [],
    "f": []
}

graph2 = {
    "0": ["8", "1", "5"],
    "1": ["0"],
    "5": ["0", "8"],
    "8": ["0", "5"],
    "2": ["3", "4"],
    "3": ["2", "4"],
    "4": ["3", "2"]
}

edges2:List[List[str]] = [
    ["w", "x"],
    ["x", "y"],
    ["z", "y"],
    ["z", "v"],
    ["w", "v"]
]

islands: List[List[str]] = [
    ["W", "L", "W", "W", "W"],
    ["W", "L", "W", "W", "W"],
    ["W", "W", "W", "L", "W"],
    ["W", "W", "L", "L", "W"],
    ["L", "W", "W", "L", "L"],
    ["L", "L", "W", "W", "W"]
]

flights_graph: dict = {
    'Mumbai': ['Paris', 'Dubai'],
    'Paris': ['Mumbai', 'Dubai', 'New York'],
    'Dubai': ['Mumbai', 'Paris', 'New York'],
    'New York': ['Paris', 'Dubai', 'Toronto'],
    'Toronto': ['New York', '']
}
