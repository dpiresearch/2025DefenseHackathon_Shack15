from collections import deque
import numpy as np
from PIL import Image
from collections import deque
import time
import random
from evaluation_env import RemoteEvaluationEnv
import sys

# What transmitter are we targeting for this run?
env = RemoteEvaluationEnv(team_id="481724", transmitter_id="tx4")
observation = env.reset()
print("Environment reset. Initial observation:", observation)

# 1) Load your mask
mp = np.asarray(Image.open('./train_data/walkable_mask.png').convert('L')) > 128

# 2) Define your 4‐neighbourhood and action codes
moves   = {0:( 0,  1),   # down
           1:( 0, -1),   # up
           2:( 1,  0),   # right
           3:(-1,  0)}   # left
opposite = {0:1, 1:0, 2:3, 3:2}

H, W = mp.shape

def in_bounds(i,j):
    return 0 <= i < W and 0 <= j < H

# BFS to build a spanning tree
def build_tree(start):
    parent = {start: None}   # maps node -> (parent_node, action_taken_to_get_here)
    q = deque([start])
    while q:
        i,j = q.popleft()
        for a,(di,dj) in moves.items():
            ni,nj = i+di, j+dj
            if in_bounds(ni,nj) and mp[nj,ni] and (ni,nj) not in parent:
                parent[(ni,nj)] = ((i,j), a)
                #print(f"Added child: {(ni,nj)} with parent: {(i,j)} and action: {a}")
                q.append((ni,nj))
    return parent

def cover_all_iter(start, parent, children, opposite):
    visited = {start}
    actions = []
    # stack holds (node, iterator over its children)
    stack = [(start, iter(children[start]))]

    while stack:
        node, child_iter = stack[-1]
        try:
            child, action = next(child_iter)
            if child not in visited:
                # step forward
                observation = env.step(action)
                # print(f"Moved in direction {action}. Observation:", observation)
                visited.add(child)
                actions.append(action)
                # dive into child's children
                #print(f"Visited child: {child} with action: {action}")
                stack.append((child, iter(children[child])))
        except StopIteration:
            # no more children → backtrack
            stack.pop()
            if parent[node] is not None:
                # parent[node] => (parent_node, action_taken_to_get_here)
                pnode, paction = parent[node]
                actions.append(opposite[paction])
    return actions

def build_children_map(parent):
    """
    parent: dict mapping node -> None (for the root) or (parent_node, action_taken)
    Returns: dict mapping each node -> list of (child_node, action_to_child)
    """
    # Initialize every node's child list
    children = {node: [] for node in parent}
    
    # For each non-root node, append it to its parent's list
    for node, val in parent.items():
        if val is not None:
            parent_node, action = val
            children[parent_node].append((node, action))
    
    return children

def find_path_to_target(start, target, parent):
    """
    Find the path from start to target using the parent map.
    Returns a list of actions to take to reach the target.

    Because of the way we build the parent map, this part
    becomes pretty easy.  We start with the target and track
    back to the start and reverse the order to get the path.
    """
    if target not in parent:
        raise ValueError("Target location is not reachable")
    
    # Reconstruct path from target to start
    path = []
    current = target
    while current != start:
        parent_node, action = parent[current]
        path.append(action)  
        current = parent_node
    # Reverse to get path from start to target
    return list(reversed(path))  

#
# Perform the walk given the start and ending cells
# We have to build the tree given the starting cell
# and then find the path to the target cell.
# We return the ending cell as the last position
# we visited to allow the next leg of the walk to be 
# constructed and executed.
#
def perform_walk(start_cell, target_cell):
    make_prediction = False
    over_threshold = False
    threshold = -100
    last_i = start_cell[0]
    last_j = start_cell[1]

    parent = build_tree(start_cell)
    path_to_target = find_path_to_target(start_cell, target_cell, parent)
    print(f"Found path to target with {len(path_to_target)} steps")
  
      # Execute the path
    for action in path_to_target:
        # Make a prediction and stop to minimize the number of steps
        if make_prediction:
            print(f"***** Making prediction at {last_i}, {last_j}")
            observation = env.step(action, (last_i, last_j, 200))
            print(f"Moved in direction {action}. Predictionbservation:", observation)
            sys.exit()
        else:   
            observation = env.step(action)
        print(f"Moved in direction {action}. Observation:", observation)

        # Check if we've crossed the threshold and if we got two
        # consecutive readings above the threshold, set the flag
        # to make a prediction.
        rssi = observation['rssi']
        if rssi > threshold and not over_threshold:
            over_threshold = True
        elif rssi > threshold and over_threshold:
            make_prediction = True
        else:
            over_threshold = False
            make_prediction = False 

        last_i = observation['ij'][0]
        last_j = observation['ij'][1]   
        time.sleep(0.1)  # Add a small delay between steps

    return (last_i, last_j)

# if your start happens to be non‐walkable, find any first mp[nj,ni]==True neighbor...
if not mp[start_cell[1], start_cell[0]]:
    raise ValueError("choose a walkable starting cell")

try:
    # The conditions of the contest are such that we always
    # start at 2219, 1812 at the beginning of every walk/reset.
    start_cell = (2219, 1812)
    target_cell = (2219, 250)
    (last_i, last_j) = perform_walk(start_cell, target_cell)
    print(f"Last position: {last_i}, {last_j}")

    target_cell = (200,500)
    (last_i, last_j) = perform_walk((last_i, last_j), target_cell)
    print(f"Last position: {last_i}, {last_j}")

    target_cell = (500,2269)
    (last_i, last_j) = perform_walk((last_i, last_j), target_cell)
    print(f"Last position: {last_i}, {last_j}")

    target_cell = (1500, 2000)
    (last_i, last_j) = perform_walk((last_i, last_j), target_cell)
    print(f"Last position: {last_i}, {last_j}")

    target_cell = (1000, 1030)
    (last_i, last_j) = perform_walk((last_i, last_j), target_cell)
    print(f"Last position: {last_i}, {last_j}")

except ValueError as e:
    print(f"Error: {e}")
