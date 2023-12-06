import numpy as np


class Graph:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def dfs(self, d, visited_vertex):
        visited_vertex[d] = True
        answer = [d]
        for i in range(len(self.matrix)):
            if self.matrix[d][i] and not visited_vertex[i]:
                answer.append(self.dfs(i, visited_vertex))
        return answer

    def fill_order(self, d, visited_vertex, stack):
        visited_vertex[d] = True
        for i in range(len(self.matrix)):
            if self.matrix[d][i] and not visited_vertex[i]:
                self.fill_order(i, visited_vertex, stack)
        stack = stack.append(d)

    def transpose(self):
        return Graph(self.matrix.T)

    def find_striongly_connected_components_count(self):
        stack = []
        visited_vertex = [False] * len(self.matrix)

        for i in range(len(self.matrix)):
            if not visited_vertex[i]:
                self.fill_order(i, visited_vertex, stack)

        gr = self.transpose()

        visited_vertex = [False] * len(self.matrix)

        components = []
        while stack:
            i = stack.pop()
            if not visited_vertex[i]:
                component = gr.dfs(i, visited_vertex)
                components.append(component)

        return len(components)
                