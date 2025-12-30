"""
Graph definitions and node role specifications for the plant layout.

This module contains:
- GRAPH_PLANT: 18-node plant layout with loop + spurs + shortcut
- NODE_ROLES_PLANT: Semantic roles for nodes (start, pickup, drop, charging)
- REWARD_PARAMS_DEFAULT: Default reward function parameters
"""

# Toy graphs for debugging
GRAPH_TOY= {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2],
}

NODE_ROLES_TOY = {
    "start": {0},
    "pickup": {1},
    "drop": {3},
}

DEFAULT_START_CANDIDATES_TOY = [0]
DEFAULT_PICKUP_CANDIDATES_TOY = [1]
DEFAULT_DROP_CANDIDATES_TOY = [3]
MAX_STEPS_TOY = 15

# Definitive graphs
GRAPH_PLANT = {
    0: [1, 10],
    1: [0, 2, 11],
    2: [1, 3, 7, 12],      # includes shortcut to 7
    3: [2, 4, 13],
    4: [3, 5],
    5: [4, 6, 14],
    6: [5, 7],
    7: [6, 8, 15, 2],      # includes shortcut to 2
    8: [7, 9, 16],
    9: [8, 10],
    10: [9, 0, 17],
    11: [1],               # spur from 1 (Workstation A)
    12: [2],               # spur from 2 (Workstation B)
    13: [3],               # spur from 3 (Assembly area)
    14: [5],               # spur from 5 (Loading bay)
    15: [7],               # spur from 7 (Storage)
    16: [8],               # spur from 8 (Packing)
    17: [10],              # spur from 10 (Charging/Maintenance)
}

NODE_ROLES_PLANT = {
    'start_candidates': [0, 6, 10],      # AGV parking/staging areas
    'pickups': [11, 12, 14],             # Material pickup locations
    'drops': [13, 15, 16],               # Delivery/drop-off locations
    'charging': [17],                    # Charging station (future use)
    'corridor': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Main loop nodes
}

REWARD_PARAMS_DEFAULT = {
    'r_step': -1.0,           # Penalty per step (encourages shorter paths)
    'r_queue_factor': -0.3,    # Queue penalty coefficient
    'r_task_completion': 50.0,     # Bonus for successful job completion
    'r_failure': -20.0,       # Penalty for failures (invalid action, timeout)
}

MAX_STEPS = 100     # Maximum steps per episode before timeout

# Default job generation candidates
DEFAULT_START_CANDIDATES = [0, 6, 10]
DEFAULT_PICKUP_CANDIDATES = [11, 12, 14]
DEFAULT_DROP_CANDIDATES = [13, 15, 16]

def validate_graph(graph):
    """
    Validate that the graph is properly formed (undirected, consistent)

    Args:
        graph: dict, adjacency lst

    Returns:
        bool: True if valid,False otherwise

    Raises:
        ValueError: If graph is invalid
    """
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in graph:
                raise ValueError(f"Node {neighbor} in adjacency list of {node} but not in graph")
            if node not in graph[neighbor]:
                raise ValueError(f"Graphs is not undirected: edge ({node}, {neighbor}) missing reverse")
    return True

def get_graph_info(graph):
    """
    Get basic statistics about the graph.

    Args:
        graph: dict, adjacency list

    Returns:
        dict: Graph statistics
    """
    num_nodes = len(graph)
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2 # undirected
    avg_degree = sum(len(neighbors) for neighbors in graph.values()) / num_nodes

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
    }

validate_graph(GRAPH_PLANT)

# Print graph info (for debugging)
if __name__ == '__main__':
    info = get_graph_info(GRAPH_PLANT)
    print("Graph Plant Statistics:")
    print(f"  Nodes: {info['num_nodes']}")
    print(f"  Edges: {info['num_edges']}")
    print(f"  Avg Degree: {info['avg_degree']:.2f}")
    print(f"\nNode Roles:")
    for role, nodes in NODE_ROLES_PLANT.items():
        print(f"  {role}: {nodes}")