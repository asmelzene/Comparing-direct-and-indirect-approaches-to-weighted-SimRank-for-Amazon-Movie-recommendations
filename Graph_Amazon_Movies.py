import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import Amazon_Movie_Parser as prs
from networkx.algorithms import bipartite

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#import datetime
#import time

# check those links
# https://books.google.se/books?id=uYttDgAAQBAJ&pg=PA159&lpg=PA159&dq=bipartite+specify+top_nodes&source=bl&ots=GJX_GzFs4t&sig=ACfU3U2O-QbIbTo7Uh3VOUlYfyEcGMvotQ&hl=en&sa=X&ved=2ahUKEwjezcag15TpAhXuxIsKHQmHBeEQ6AEwA3oECAgQAQ#v=onepage&q&f=false

# https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.bipartite.html

# https://pynetwork.readthedocs.io/en/latest/networkx_basics.html

# https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.components.connected.connected_component_subgraphs.html

# https://kite.com/python/docs/networkx.reportviews.EdgeDataView

# https://stackoverflow.com/questions/24829123/plot-bipartite-graph-using-networkx-in-python

class Graph_Amazon:
    def __init__(self):
        #self.label_size = 10
        #self.image_size = 32 * 32 * 3
        #for one_hot_encoding
        #self.label_encoder = preprocessing.LabelBinarizer()
        # file path of the images on your laptop
        #self.filePath = 'Dataset/data_batch_1'
        #self.unique_labels = []
        self.file_name = 'movies.txt'
        
    def Create_and_Draw_Bipartite(self, file_name = 'movies.txt', n_movies = 200, prs_out = 'dictionary'):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        movie_dict = prs.AmazonMovies(n_movies, file_name, prs_out)

        movie_name = []
        userId = []
        edges_U_M = []

        gr_amazon_movies = nx.Graph()
        gr_amazon_movies.add_nodes_from(userId, bipartite=0)
        gr_amazon_movies.add_nodes_from(movie_name, bipartite=1)
        #gr_amazon_movies.add_edges_from(edges_U_M)

        for movie in movie_dict:
            gr_amazon_movies.add_edge(movie_dict.get(movie)[1], movie_dict.get(movie)[0], weight=movie_dict.get(movie)[3])

        # connected_component_subgraphs >> deprecated
        #max_connected_gr_amazon_movies = max(nx.connected_component_subgraphs(gr_amazon_movies), key=len)
        max_connected_gr_amazon_movies = max((gr_amazon_movies.subgraph(c) for c in nx.connected_components(gr_amazon_movies)), key=len)
        
        self.bottom_nodes, self.top_nodes = bipartite.sets(max_connected_gr_amazon_movies)

        #self.top = nx.bipartite.sets(max_connected_gr_amazon_movies)[0]
        pos = nx.bipartite_layout(max_connected_gr_amazon_movies, self.top_nodes)
        nx.draw(max_connected_gr_amazon_movies, pos, with_labels=1)
        nx.draw_networkx_edge_labels(max_connected_gr_amazon_movies, pos)
        #nx.is_connected(G)

        plt.show()
        
        return max_connected_gr_amazon_movies
    
    def Create_Bipartite(self, file_name = 'movies.txt', n_movies = 200, prs_out = 'dictionary'):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        if prs_out == 'dictionary':
            movie_dict = prs.AmazonMovies(n_movies, file_name, prs_out)

            #movie_name = []
            #userId = []
            
            gr_amazon_movies = nx.Graph()
            #gr_amazon_movies.add_nodes_from(userId, bipartite=0)
            #gr_amazon_movies.add_nodes_from(movie_name, bipartite=1)

            # userId, productId, score
            for movie in movie_dict:
                print(movie_dict.get(movie))
                # usedId__(movie)[1] > movieId__(movie)[0]
                gr_amazon_movies.add_edge(movie_dict.get(movie)[1], movie_dict.get(movie)[0], weight=movie_dict.get(movie)[3])

            self.gr_amazon_movies = gr_amazon_movies
                
            # connected_component_subgraphs >> deprecated
            #max_connected_gr_amazon_movies = max(nx.connected_component_subgraphs(gr_amazon_movies), key=len)
            max_connected_gr_amazon_movies = max((gr_amazon_movies.subgraph(c) for c in nx.connected_components(gr_amazon_movies)), key=len)

            self.bottom_nodes, self.top_nodes = bipartite.sets(max_connected_gr_amazon_movies)

            return max_connected_gr_amazon_movies
        elif prs_out == 'file' or prs_out == 'screen':
            prs.AmazonMovies(n_movies, file_name, prs_out)
    
    def Draw_Bipartite(self, max_connected_gr_amazon_movies):
        #self.top = nx.bipartite.sets(max_connected_gr_amazon_movies)[0]
        pos = nx.bipartite_layout(max_connected_gr_amazon_movies, self.top_nodes)
        nx.draw(max_connected_gr_amazon_movies, pos, with_labels=1)
        nx.draw_networkx_edge_labels(max_connected_gr_amazon_movies, pos)

        plt.show()
        
    def Show_Nodes(self):
        print('Sum-Top Nodes:{}\nTop nodes (movies): {}'.format(len(self.top_nodes), self.top_nodes))
        print('Sum-Bottom Nodes:{}\nBottom nodes (users):\n {}'.format(len(self.bottom_nodes), self.bottom_nodes))
    
    def Create_Graph_From_List_No_Weight(self, movie_list):
        userId = []
        movie_name = []
            
        for movie in movie_list:
            userId.append(movie[0])
            movie_name.append(movie[1])
            
        gr_amazon_movies = nx.Graph() 
        gr_amazon_movies.add_nodes_from(userId, bipartite=0)
        gr_amazon_movies.add_nodes_from(movie_name, bipartite=1)
        
        return gr_amazon_movies

    def Create_Graph_From_List_WITH_Weight(self, movie_list):
        userId = []
        movie_name = []
            
        gr_amazon_movies = nx.Graph() 

        for movie in movie_list:
            gr_amazon_movies.add_edge(movie[0], movie[1], weight=movie[2])
            #userId.append(movie[0])
            #movie_name.append(movie[1])
        
        return gr_amazon_movies
        
    def Get_All_Edges_with_Weight(self, max_connected_gr_amazon_movies):
        #lst_x=list(max_connected_gr_amazon_movies.edges.data())
        #max_connected_gr_amazon_movies.edges('AMEJTGF5NQL')
        #edge_view_grp = max_connected_gr_amazon_movies.edges.data('weight', default=1)
        #edge_view_grp[0]
        # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.edges.html
        edge_list = list(max_connected_gr_amazon_movies.edges.data('weight', default=1))
        return edge_list
    
    def Is_Connected_Bipartite(self, max_connected_gr_amazon_movies):
        print('Is connected: {}'.format(nx.is_connected(max_connected_gr_amazon_movies)))
        print('Is bipartite: {}'.format(nx.is_bipartite(max_connected_gr_amazon_movies)))