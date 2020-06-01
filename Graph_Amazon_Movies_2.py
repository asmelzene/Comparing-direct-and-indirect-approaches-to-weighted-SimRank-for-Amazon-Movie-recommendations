import Amazon_Movie_Parser as prs

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from datetime import datetime

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
    
    def Create_Graph(self, file_name = 'movies.txt', n_movies = 200, prs_out = 'dictionary'):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        # here, we don't care ig the graph is CONNECTED or NOT
        if prs_out == 'dictionary':
            movie_dict = prs.AmazonMovies(n_movies, file_name, prs_out)

            gr_amazon_movies = nx.Graph()

            # userId, productId, score
            for movie in movie_dict:
                #print(movie_dict.get(movie))
                # usedId__(movie)[1] > movieId__(movie)[0]
                gr_amazon_movies.add_edge(movie_dict.get(movie)[1], movie_dict.get(movie)[0], weight=movie_dict.get(movie)[3])

            self.gr_amazon_movies = gr_amazon_movies

            return gr_amazon_movies
        
        elif prs_out == 'file' or prs_out == 'screen':
            prs.AmazonMovies(n_movies, file_name, prs_out)
            
    def Create_Bipartite_BEFORE_DATE(self, file_name = 'movies.txt', n_movies = 500, n_movies_val = 205, prs_out = 'dictionary', file_type = 'txt'):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        if prs_out == 'dictionary':
            if file_type == 'txt':
                movie_dict = prs.AmazonMovies(n_movies + n_movies_val, file_name, prs_out)
            elif file_type == 'pickle':
                movie_dict = prs.Load_Pickle_File(file_name, n_movies + n_movies_val)

            lst_movie = []
            lst_user = []
            lst_rating = []
            lst_time = []    
            
            for review in movie_dict.values():
                ts = int(review[2])
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')

                lst_movie.append(review[0])
                lst_user.append(review[1])
                #lst_time.append(review[2])
                lst_time.append(ts_datetime)
                lst_rating.append(review[3])
                
            dict_reviews = {'users': lst_user, 'movies': lst_movie, 'rating': lst_rating, 'time': lst_time}

            df_reviews = pd.DataFrame(dict_reviews).sort_values('time')
            df_reviews = df_reviews.reset_index(drop=True)
            
            df_reviews_model = df_reviews.query("index < " + str(n_movies))
            df_reviews_val = df_reviews.query("index >= " + str(n_movies) + " and index < " + str(n_movies + n_movies_val))
            
                
            gr_amazon_movies = nx.Graph()
            gr_amazon_movies_val = nx.Graph()
            
            # userId, productId, score
            for review in df_reviews_model.values:
                #print(movie_dict.get(movie))
                # =review[0]=usedId ... review[1]=movieId ... review[2]=rating ... review[3]=time
                gr_amazon_movies.add_edge(review[0], review[1], weight = review[2])
                
            for review in df_reviews_val.values:
                #print(movie_dict.get(movie))
                # =review[0]=usedId ... review[1]=movieId ... review[2]=rating ... review[3]=time
                gr_amazon_movies_val.add_edge(review[0], review[1], weight = review[2])

            self.gr_amazon_movies = gr_amazon_movies
            self.gr_amazon_movies_val = gr_amazon_movies_val
                
            # connected_component_subgraphs >> deprecated
            #max_connected_gr_amazon_movies = max(nx.connected_component_subgraphs(gr_amazon_movies), key=len)
            max_connected_gr_amazon_movies = max((gr_amazon_movies.subgraph(c) for c in nx.connected_components(gr_amazon_movies)), key=len)
            
            max_connected_gr_amazon_movies_val = max((gr_amazon_movies_val.subgraph(c) for c in nx.connected_components(gr_amazon_movies_val)), key=len)

            self.bottom_nodes, self.top_nodes = bipartite.sets(max_connected_gr_amazon_movies)
            self.bottom_nodes_v, self.top_nodes_v = bipartite.sets(max_connected_gr_amazon_movies_val)

            return max_connected_gr_amazon_movies, max_connected_gr_amazon_movies_val
        
        elif prs_out == 'file' or prs_out == 'screen':
            prs.AmazonMovies(n_movies, file_name, prs_out)
            
    def Create_Bipartite(self, file_name = 'movies.txt', num_limit=500000, n_movies = 500, n_movies_val = 205, prs_out = 'dictionary', file_type = 'txt', date_year=2009):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        if prs_out == 'dictionary':
            if file_type == 'txt':
                movie_dict = prs.AmazonMovies(num_limit, n_movies + n_movies_val, date_year, file_name, prs_out)
            elif file_type == 'pickle':
                movie_dict = prs.Load_Pickle_File(file_name, n_movies + n_movies_val)

            lst_movie = []
            lst_user = []
            lst_rating = []
            lst_time = []    
            
            for review in movie_dict.values():
                #ts = int(review[2])
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')

                lst_movie.append(review[0])
                lst_user.append(review[1])
                lst_time.append(review[2])
                #lst_time.append(ts_datetime)
                lst_rating.append(review[3])
                
            dict_reviews = {'users': lst_user, 'movies': lst_movie, 'rating': lst_rating, 'time': lst_time}

            df_reviews = pd.DataFrame(dict_reviews).sort_values('time')
            df_reviews = df_reviews.reset_index(drop=True)
            
            df_reviews_model = df_reviews.query("index < " + str(n_movies))
            df_reviews_val = df_reviews.query("index >= " + str(n_movies) + " and index < " + str(n_movies + n_movies_val))
            
                
            gr_amazon_movies = nx.Graph()
            gr_amazon_movies_val = nx.Graph()
            
            # userId, productId, score
            for review in df_reviews_model.values:
                #print(movie_dict.get(movie))
                # =review[0]=usedId ... review[1]=movieId ... review[2]=rating ... review[3]=time
                gr_amazon_movies.add_edge(review[0], review[1], weight = review[2])
                
            for review in df_reviews_val.values:
                #print(movie_dict.get(movie))
                # =review[0]=usedId ... review[1]=movieId ... review[2]=rating ... review[3]=time
                gr_amazon_movies_val.add_edge(review[0], review[1], weight = review[2])

            self.gr_amazon_movies = gr_amazon_movies
            self.gr_amazon_movies_val = gr_amazon_movies_val
                
            # connected_component_subgraphs >> deprecated
            #max_connected_gr_amazon_movies = max(nx.connected_component_subgraphs(gr_amazon_movies), key=len)
            max_connected_gr_amazon_movies = max((gr_amazon_movies.subgraph(c) for c in nx.connected_components(gr_amazon_movies)), key=len)
            
            max_connected_gr_amazon_movies_val = max((gr_amazon_movies_val.subgraph(c) for c in nx.connected_components(gr_amazon_movies_val)), key=len)

            self.bottom_nodes, self.top_nodes = bipartite.sets(max_connected_gr_amazon_movies)
            self.bottom_nodes_v, self.top_nodes_v = bipartite.sets(max_connected_gr_amazon_movies_val)

            return max_connected_gr_amazon_movies, max_connected_gr_amazon_movies_val
        
        elif prs_out == 'file' or prs_out == 'screen':
            prs.AmazonMovies(n_movies, file_name, prs_out)
    
    def Create_Bipartite__x(self, file_name = 'movies.txt', n_movies = 200, prs_out = 'dictionary', file_type = 'txt'):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        if prs_out == 'dictionary':
            if file_type == 'txt':
                movie_dict = prs.AmazonMovies(n_movies, file_name, prs_out)
            elif file_type == 'pickle':
                movie_dict = prs.Load_Pickle_File(file_name, n_movies)

            #movie_name = []
            #userId = []
            
            gr_amazon_movies = nx.Graph()
            #gr_amazon_movies.add_nodes_from(userId, bipartite=0)
            #gr_amazon_movies.add_nodes_from(movie_name, bipartite=1)

            # userId, productId, score
            for movie in movie_dict:
                #print(movie_dict.get(movie))
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
            
    def Create_Bipartite_VALIDATION(self, file_name = 'movies.txt', start_index = 40000, n_movies = 200, prs_out = 'dictionary', file_type = 'txt'):
        #prs_out = 'dictionary'  # options: screen, file.  # file name is determined automatically if file is chosen
        if prs_out == 'dictionary':
            print('Create_Bipartite_VALIDATION is running...')
            if file_type == 'txt':
                movie_dict = prs.AmazonMovies_VALIDATION(n_movies, file_name, prs_out, start_index)
            elif file_type == 'pickle':
                movie_dict = prs.Load_Pickle_File_VAL(file_name, start_index, n_movies)

            #movie_name = []
            #userId = []
            
            gr_amazon_movies = nx.Graph()
            #gr_amazon_movies.add_nodes_from(userId, bipartite=0)
            #gr_amazon_movies.add_nodes_from(movie_name, bipartite=1)

            # userId, productId, score
            for movie in movie_dict:
                #print(movie_dict.get(movie))
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
    
    def Create_Graph_From_List_NO_EDGE(self, movie_list):
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
        i_c = nx.is_connected(max_connected_gr_amazon_movies)
        i_b = nx.is_bipartite(max_connected_gr_amazon_movies)
        print('Is connected: {} ... Is bipartite: {}'.format(i_c, i_b))