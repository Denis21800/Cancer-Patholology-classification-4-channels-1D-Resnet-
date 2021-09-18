import networkx as nx
import matplotlib.pyplot as plt


class NetGraph(object):
    def __init__(self, matrix, labels):
        self.graph = nx.from_numpy_matrix(matrix, create_using=nx.Graph)
        #  self.graph = nx.relabel_nodes(self.graph, labels)

        pos = nx.spring_layout(self.graph,
                               weight='weight',
                               iterations=100)
        nx.draw(self.graph,
                pos=pos,
                cmap=plt.get_cmap('jet'),
                node_color=list(labels.keys()),
                node_size=800,
                with_labels=True)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        for e in edge_labels:
            w = edge_labels.get(e)
            if e != 0:
                w = round(1 / w, 2)
            edge_labels.update({e: w})
        nx.draw_networkx_edge_labels(self.graph,
                                     pos=pos,
                                     edge_labels=edge_labels)

        plt.savefig('./graph/graph.png')
