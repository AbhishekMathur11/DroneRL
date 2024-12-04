import heapq

class Node:
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def heuristic(node, goal):
    return abs(node.x - goal[0]) + abs(node.y - goal[1])

def a_star(start, goal, grid):
    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1], 0)
    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if (current_node.x, current_node.y) == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = current_node.x + i, current_node.y + j

            if 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and grid[next_x][next_y] == 0 and (next_x, next_y) not in closed_set:
                next_node = Node(next_x, next_y, current_node.cost + 1, current_node)
                heapq.heappush(open_set, next_node)

    return None

# Example usage:
start_point = (0, 0)
goal_point = (4, 4)
obstacle_grid = [[0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0]]

path_a_star = a_star(start_point, goal_point, obstacle_grid)
print("A* Path:", path_a_star)
