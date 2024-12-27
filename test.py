
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class GraphDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def update(self, value):
        self.fig.delaxes(self.ax)    
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(value)
        plt.pause(0.1)

class DataGenerator:
    def __init__(self, graph_drawer):
        self.graph_drawer = graph_drawer

    def generate_data(self):
        for i in range(200):

            self.graph_drawer.update(np.random.rand(10))
            

if __name__ == "__main__":
    graph = GraphDrawer()
    data_generator = DataGenerator(graph)
    data_generator.generate_data()