import random
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def rrt(start, goal, obstacle_func, max_iter=1000, step_size=1):
    tree = [Node(start[0], start[1])]

    for _ in range(max_iter):
        rand_node = Node(random.uniform(0, 10), random.uniform(0, 10))  # Adjust the bounds based on your environment

        nearest_node = min(tree, key=lambda n: (n.x - rand_node.x)**2 + (n.y - rand_node.y)**2)

        if obstacle_func(nearest_node, rand_node, step_size):
            new_node_x = nearest_node.x + step_size * (rand_node.x - nearest_node.x)
            new_node_y = nearest_node.y + step_size * (rand_node.y - nearest_node.y)

            if obstacle_func(Node(nearest_node.x, nearest_node.y), Node(new_node_x, new_node_y), step_size):
                tree.append(Node(new_node_x, new_node_y))

                if (new_node_x, new_node_y) == goal:
                    return tree

    return None

def obstacle_free(node1, node2, step_size):
    # Implement obstacle checking logic based on your environment
    return True

# Example usage:
start_point_rrt = (0, 0)
goal_point_rrt = (9, 9)

rrt_path = rrt(start_point_rrt, goal_point_rrt, obstacle_free)
print("RRT Path:", [(node.x, node.y) for node in rrt_path])

# Plotting the RRT path
plt.scatter(*zip(*[(node.x, node.y) for node in rrt_path]), color='blue')
plt.plot(*zip(*[(node.x, node.y) for node in rrt_path]), color='blue')
plt.scatter(*start_point_rrt, color='green', marker='s', label='Start')
plt.scatter(*goal_point_rrt, color='red', marker='s', label='Goal')
plt.legend()
plt.show()
