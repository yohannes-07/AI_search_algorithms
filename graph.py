from collections import defaultdict, deque
from math import sqrt
from queue import PriorityQueue
from random import randint
from timeit import repeat

from docx import Document

# Adjacency list
# each node have node and neighbours
class Node:
    def __init__(self, val, neighbours) -> None:
        self.val = val
        self.neighbours = neighbours

# we used defaultdict not to wirte many if else conditions(more readable code)       
class Graph:
    def __init__(self) -> None:
        self.adj_list = defaultdict(list)

        # latitude and longtiude location for each cities
        self.location = {}
    
   
    def createNode(self, val, neighbors = []):
        if len(neighbors)==0:
            neighbors = []
        return Node(val, neighbors)
    
    # add new node with node and neighbours list(set to empty list if neighbours aren't provided)
    def addNode(self, node):
        self.adj_list[node.val] = node.neighbours
        
    def addNodes(self, nodes):
        for node in nodes:
            self.addNode(node)

    # remove node
    def removeNode(self, node):
        self.adj_list.pop(node) 
    
    # add neighbours and make connection between them
    def addNeighbour(self, node, neighbour, weight = 1):
        if (neighbour, weight) in self.adj_list[node]:
            return
       
        self.adj_list[node].append((neighbour, weight))
        
        self.adj_list[neighbour].append((node, weight))
    
    # add neighbours to a node

    def addNeighbours(self, node, neighbours = []):
        self.adj_list[node].extend(neighbours)
        
        for neighbour, weight in neighbours:
            self.adj_list[neighbour].append((node, weight))

    # remove neighbours(connections) of a node O(n) time complexity
    def removeNeighbour(self, node, neighbour, weight):
        if node in self.adj_list:

            try:
                self.adj_list[node].remove((neighbour, weight))
                self.adj_list[neighbour].remove((node, weight))

            except ValueError:
                print("neighbour doesn't exist")

    # get all neighbours(connections) of a node
    def getConnections(self, node):
        return self.adj_list[node]

    # get the entire graph
    def printGraph(self):
        print(self.adj_list)
        return 
    
    #check if the node exists in the graph
    def isNodeExists(self, node):
        return node in self.adj_list
    

#  ALGORITHMS

    # BFS
    def BFS(self, start, goal):
        visted = {start}
        queue  = deque([(start, [start])])

        while queue:
            curr, path = queue.popleft()
            if curr == goal:
                return path

            for neighbour, _ in self.adj_list[curr]:

                if neighbour not in visted:
                    visted.add(neighbour)
                    queue.append((neighbour, path +  [neighbour]))

        return "Unreachable"

    def DFS(self, start, goal,):

        path = [start]
        visted = set()

        def dfs(start, goal):
            if start not in visted:
                visted.add(start)

                if start == goal:
                    return True
                
                for neighbour, cost in self.adj_list[start]:

                    path.append(neighbour)
                    isFound = dfs(neighbour, goal)

                    if isFound: return True

                    path.pop()
             
        if dfs(start, goal):
            return path

        return "Unreachable"


    def UCS(self, start, goal):
            queue = PriorityQueue()
            queue.put((0, start, []))

            visted = set()

            while not queue.empty():
                cost, node, path = queue.get()

                if goal == node:
                    # path = " => ".join(path + [node])
                    return f'total cost: {cost}   path: {path + [node]}'

                
                visted.add(node)

                for neighbour, edge_cost in self.adj_list[node]:
                    if neighbour in visted:
                        continue
                    
                    total_cost = cost + edge_cost
                    queue.put((total_cost, neighbour, path + [node]))

            return "No path to reach to {goal}".format(goal = goal)


    def iterativeDeepening(self,start, goal, maxDepth):

        def DLS(currNode, currDepth, path, visted ):

            if currNode == goal:
                path.append(currNode)
                return True
            
            if currDepth <= 0:
                return False
            
            visted.add(currNode)

            for neighbour, edge_cost in self.adj_list[currNode]:
                if neighbour in visted:
                    continue

                path.append(currNode)

                if DLS(neighbour, currDepth - 1, path, visted):
                    return True
                
                path.pop()


        for currDepth in range(maxDepth):
            visted, path = set(), []
            if DLS(start, currDepth, path, visted):
                return path
            
        return "Unreachable"


    def bidirectionalSearch(self, start, goal):
        forward_visited, backward_visited = {start}, {goal}

        forwardQueue, backwardQueue = PriorityQueue(), PriorityQueue()

        forwardQueue.put((0, start))  # (cost, node)
        backwardQueue.put((0, goal))  # (cost, node)

        from_start = {start: None}
        from_goal = {goal: None}

        while not forwardQueue.empty() and not backwardQueue.empty():

            if forwardQueue.qsize() <= backwardQueue.qsize():
                _, current = forwardQueue.get()

                if current in from_goal:
                    return self.path(from_start, neighbor, from_goal, start, goal)
                
              
                for neighbor, cost in self.adj_list[current]:
                    if neighbor not in forward_visited:

                        forward_visited.add(neighbor)
                        forwardQueue.put((cost, neighbor))

                        from_start[neighbor] = current

            else:
                _, current = backwardQueue.get()

                if current in from_start:
                    return self.path(from_start, neighbor, from_goal, start, goal)
               
                for neighbor, cost in self.adj_list[current]:
                    if neighbor not in backward_visited:

                        backward_visited.add(neighbor)
                        backwardQueue.put((cost, neighbor))
                        from_goal[neighbor] = current

        return "They aren't connected"


    def path(self, from_start, shared_node, from_goal, start, goal):
        path = [shared_node]
        node = shared_node

        while node != start:
            node = from_start[node]
            path.append(node)

        path = path[::-1]
        print(path)

        node = shared_node
        while node != goal:
            node = from_goal[node]
            path.append(node)

        return path


    def greedySearch(self, start, goal):

        queue = PriorityQueue()
        visted = set()

        heuristic_cost = self.heuristic(start, goal)
        queue.put((heuristic_cost, start))

        path = {start: None}

        while not queue.empty():
            _, node = queue.get()

            if node == goal:
                answerPath = []
                while node:
                    answerPath.append(node)
                    node = path[node]

                answerPath.reverse()
                return answerPath
            
            visted.add(node)

            for neighbour, _ in self.adj_list[node]:
                if neighbour in visted:
                    continue
                
                queue.put(( self.heuristic(neighbour, goal), neighbour))
                path[neighbour] = node
               

        return " Unreachable"
    
    def astar(self, start, goal):

        queue = PriorityQueue()
        visted = set()

        heuristic_cost = self.heuristic(start, goal)
        queue.put((0 + heuristic_cost, start))

        total_cost = {start: 0}
        path = {start: None}

        while not queue.empty():
            curr_cost, node = queue.get()

            if node == goal:
                answerPath = []
                while node:
                    answerPath.append(node)
                    node = path[node]

                answerPath.reverse()
                return answerPath
            
            visted.add(node)

            for neighbour, edge_cost in self.adj_list[node]:
                if neighbour in visted and  curr_total_cost >= total_cost[neighbour]:
                    continue


                curr_total_cost = total_cost[node] + edge_cost

                # not evaluated yet but put in queue
                pending = [node for cost, node in queue.queue]


                if neighbour not in pending:
                    queue.put((curr_total_cost + self.heuristic(neighbour, goal), neighbour))

                elif curr_total_cost >= total_cost[neighbour]:
                    continue

                path[neighbour] = node
                total_cost[neighbour] = curr_total_cost

        return " Unreachable"
    
    
    def heuristic(self, start, goal):
        return sqrt((self.location[start][0] - self.location[goal][0]) ** 2 + (self.location[start][1] - self.location[goal][1]) ** 2)

        
    def fileReader(self, fileName):
        with open(fileName, 'r') as file:
            for line in file:
                city, latitude, longtidue = line.strip().split("    ")
                self.location[city] = (float(latitude,), float(longtidue))

graph = Graph()

graph.fileReader("absoluteLocation.txt")

Arad = graph.createNode('Arad', [('Sibiu', 140), ('Timisoara', 118), ('Zerind',75 )])
Sibu = graph.createNode('Sibiu', [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)])
Zerind = graph.createNode('Zerind', [('Oradea', 71), ('Arad',75 )])
Timisora = graph.createNode('Timisoara', [('Arad', 118), ('Lugoj', 111)])
Lugoj =  graph.createNode('Lugoj', [('Timisoara', 111), ('Mehadia', 70)])
Mehadia = graph.createNode('Mehadia', [('Lugoj', 70), ('Drobeta', 75)])
Drobeta = graph.createNode('Drobeta', [('Mehadia', 75), ('Craiova', 120)])
Craiova = graph.createNode('Craiova', [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)])
Rimnicu = graph.createNode('Rimnicu Vilcea', [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)])
Oradea = graph.createNode('Oradea', [('Sibiu', 151),('Zerind', 71) ])
Fagaras = graph.createNode('Fagaras', [('Sibiu', 99), ('Bucharest', 211)])
Pitesti = graph.createNode('Pitesti', [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)])
Bucharest = graph.createNode('Bucharest', [('Fagaras', 211), ('Pitesti', 101), ('Urziceni', 85), ('Giurgiu', 90)])
Urziceni = graph.createNode('Urziceni', [('Vaslui', 142), ('Hirsova', 98), ('Bucharest', 85)])
Hirsova = graph.createNode('Hirsova', [('Eforie', 86), ('Urziceni', 98)])
Eforie = graph.createNode('Eforie', [('Hirsova', 86)])
Vaslui = graph.createNode('Vaslui', [('Iasi', 92), ('Urziceni', 142)])
Iasi = graph.createNode('Iasi', [('Vaslui', 92), ('Neamt', 87)])
Neamt = graph.createNode('Neamt', [('Iasi', 87)])
Giurgiu = graph.createNode('Giurgiu', [('Bucharest', 90)])

nodes = [Eforie, Neamt, Iasi ,Giurgiu, Arad, Sibu, Zerind, Timisora, Lugoj, Mehadia, Drobeta, Craiova, Rimnicu, Oradea, Fagaras, Pitesti, Bucharest, Urziceni, Hirsova, Vaslui, Iasi]
for node in nodes:
    graph.addNode(node)

# select 10 random random cities
cities = [city for city in graph.adj_list.keys()]
random_cities = []
for i in range(10):
    while True:
        random_city = cities[randint(0, len(cities)-1)]
        if random_city not in random_cities:
            random_cities.append(random_city)
            break

# print(graph.astar('Bucharest', 'Arad'))


set_up = '''from __main__ import Graph
from collections import defaultdict, deque
from math import sqrt
from queue import PriorityQueue
from random import randint'''

bfs_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            graph.bfs(random_cities[i], random_cities[j])
'''

dfs_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            graph.dfs(random_cities[i], random_cities[j])
'''

astar_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            graph.astar(random_cities[i], random_cities[j])
'''

ucs_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            graph.UCS(random_cities[i], random_cities[j])
'''

iter_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            graph.iterativeDeepening(random_cities[i], random_cities[j])
'''

greedy_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            graph.greedySearch(random_cities[i], random_cities[j])
'''


all_paths = []
# for dfs
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(graph.DFS(random_cities[i], random_cities[j]))-1
all_paths.append(p)
# for A* and UCS
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(graph.astar(random_cities[i], random_cities[j]))-1
all_paths.append(p)
all_paths.append(p)
# for iterativeDeepening
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(graph.iterativeDeepening(random_cities[i], random_cities[j], 5))-1
all_paths.append(p)
# for BFS
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(graph.BFS(random_cities[i], random_cities[j]))-1
all_paths.append(p)
# for greedy
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(graph.greedySearch(random_cities[i], random_cities[j]))-1
all_paths.append(p)


# create word doc and record the time and path length
document = Document()

table = document.add_table(rows=1, cols=7)
hdr_cells = table.rows[0].cells

codes = [dfs_code, ucs_code, astar_code, iter_code, bfs_code, greedy_code]
alg_names = ['DFS', 'UCS', 'A*', 'Iterative', 'BFS', 'Greedy']

hdr_cells[0].text = ''
for i in range(len(alg_names)):
    hdr_cells[i+1].text = alg_names[i]

all_ts = []
t_avs = []
rep = 10
for code in codes:
    each = repeat(setup= set_up,
                    stmt= code,
                    repeat= rep,
                    number=1)
    all_ts.append(each)
    t_av = sum(each) / rep
    t_avs.append(t_av)

for i in range(10):
    row_cells = table.add_row().cells
    row_cells[0].text = 'Trial '+str(i+1)
    for j in range(6):
        row_cells[j+1].text = str(all_ts[j][i])

row_cells = table.add_row().cells
row_cells[0].text = 'Average time'
for i in range(6):
    row_cells[i+1].text = str(t_avs[i])

row_cells = table.add_row().cells
row_cells[0].text = 'Path Length'
for i in range(6):
    row_cells[i+1].text = str(all_paths[i])

document.save('cities.docx')

