import Graph_Amazon_Movies as gam
import Amazon_Movie_Parser as amp
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
import networkx as nx
import Amazon_Movie_Parser as prs
import datetime
from numpy import savetxt

class GraphComp:
    def __init__(self, debug_mode = 'Off'):
        self.debug_mode = debug_mode
    
    def load_data_movies():
        try:
            # needed only if you don't have the movies.txt file in your local 
            # or if you use colab 
            from google_drive_downloader import GoogleDriveDownloader as gdd
            file_id1='1iytA1n2z4go3uVCwE__vIKouTKyIDjEq'
            # movies.txt >> 9 GB data  ... below shared URL has the fileID to download
            #https://drive.google.com/file/d/1cRLjG6Pl4JEag-utmD-7nLipzxCdmkiQ/view?usp=sharing
            file_id2='1cRLjG6Pl4JEag-utmD-7nLipzxCdmkiQ'
            gdd.download_file_from_google_drive(file_id=file_id2,
                                                dest_path='data/movies.txt',
                                                unzip=True)
        except:
            print('no need for laptop')
        
    def Create_Bipartite_Giant_Component(self, grp, file_name='data/movies.txt', n_movies=40000, prs_out='dictionary', file_type='txt'):
        # grp: Graph_Amazon_Movies.Graph_Amazon()
        start_time = datetime.datetime.now()
        
        max_connected_gr_amazon_movies = grp.Create_Bipartite(file_name, n_movies, prs_out, file_type)

        #bottom_nodes, top_nodes = bipartite.sets(max_connected_gr_amazon_movies)
        end_time = datetime.datetime.now()
        print('Calculation time-Crete graph from the movies file: {}'.format(end_time-start_time))

        return max_connected_gr_amazon_movies
    
    def Create_Bipartite_Giant_Component_VAL(self, grp, file_name='data/movies.txt', start_index=40000, n_movies=5000, prs_out='dictionary', file_type='txt'):
        start_time = datetime.datetime.now()
        # we can save train and validation data in files so that we don't need to read movies.txt each time

        max_connected_gr_amazon_movies = grp.Create_Bipartite_VALIDATION(file_name, start_index, n_movies, prs_out, file_type)

        #bottom_nodes, top_nodes = bipartite.sets(max_connected_gr_amazon_movies)
        end_time = datetime.datetime.now()
        print('Calculation time-Crete graph from the movies file: {}'.format(end_time-start_time))

        return max_connected_gr_amazon_movies
    
    def health_check(self, grp, max_connected_gr_amazon_movies, bottom_nodes, top_nodes):
        # Data check
        # some MOVIES don't start with B but instead full numbers like: 0790747324
        if self.debug_mode == 'On':
            print('max_connected_gr_amazon_movies:')
        count_B = 0
        for movie in top_nodes:
            if movie.startswith('B'):
                count_B += 1

        if self.debug_mode == 'On':
            print('Movies: {} ... starts with B: {}'.format(len(top_nodes), count_B))

        count_A=0
        for movie in bottom_nodes:
            if movie.startswith('A'):
                count_A += 1

        if self.debug_mode == 'On':
            print('Users: {} ... starts with A: {}'.format(len(bottom_nodes), count_A))

        grp.Is_Connected_Bipartite(max_connected_gr_amazon_movies)
        # edges between movies
        #max_grp_top = bipartite.projected_graph(max_connected_gr_amazon_movies, grp.top_nodes)
        # shows all the records BEFORE picking up the MAX CONNECTED component - most likely it will be a DISCONNECTED GRAPH
        # returns a LIST
        #edge_list_ALL = grp.Get_All_Edges_with_Weight(grp.gr_amazon_movies)
        #print(edge_list_ALL[0:5])
        #edge_list_connected = grp.Get_All_Edges_with_Weight(max_connected_gr_amazon_movies)
        #list(max_connected_gr_amazon_movies.edges)[0:5]
        # shows the nodes of the max connected component/subgraph
        #grp.Show_Nodes()
        #if your graph is too BIG, it is better not to draw it, it takes time..
        #grp.Draw_Bipartite(max_connected_gr_amazon_movies)
        #shows the edges with weights
        #list(max_connected_gr_amazon_movies.edges.data('weight', default=1))[0:5]

        # double check if the numbers are matching before saving the records
        # and after creating the graph from the saved file
        print('# of nodes: {} ... # of edges: {}'.format(len(max_connected_gr_amazon_movies.nodes), \
                                                                 len(max_connected_gr_amazon_movies.edges)))

        ##### TEST 
        #print('AFTER: # of nodes: {} ... # of edges: {}'.format(len(grp_giant.nodes), len(grp_giant.edges)))

        # this is to fetch all nodes without edges, so that we can use them to predict the edges

        ##### TEST 
        #grp_giant_no_edges = grp.Create_Graph_From_List_NO_EDGE(grp_giant)
        #if you check it, you'll see no edges but nodes
        #grp_giant_no_edges.edges
        #list(grp_giant_no_weights.nodes)[0:5]
        #grp.Is_Connected_Bipartite(grp_giant_no_weights)
        
    def Normalize_Array(self, arr_to_normalize, axis=0):
        if axis == 1:
            arr_sum_per_row = arr_to_normalize.sum(axis=1)
            arr_norm = arr_to_normalize/np.tile(arr_sum_per_row, (arr_sum_per_row.shape[0], 1)).T
            # >> this works with tt sample
        elif axis == 0:
            arr_sum_per_row = arr_to_normalize.sum(axis=0)
            arr_norm = arr_to_normalize/np.tile(arr_sum_per_row, (arr_sum_per_row.shape[1], 1)).T
            # >> this works with MOVIES >> I need to check why...
        return arr_norm

        #tt = np.array([[2, 4, 6], [1, 5, 8], [20, 4, 6]])
        # normalize P values (between 0 and 1) so that they can reflect probabilities 
        # the probability with itself will be the highest
        # !!! IMPORTANT !!!
        # the probability with movie nodes will be ignored since we are looking for the relations(similarity-measure) 
        # with other users
        #P_norm = np.round(Normalize_Array(P), 6)
        
    def Normalize_Matrix(self, arr_to_normalize, dim = 'row'):
        #tt = np.array([[2, 4, 6], [1, 5, 8], [20, 4, 6]])
        arr_row_sums = arr_to_normalize.sum(axis=1)

        if dim == 'row':
            arr_norm = arr_to_normalize / arr_row_sums[np.newaxis][0]
            # >> this one works for MOVIES
        else:
            arr_norm = arr_to_normalize / arr_row_sums[:, np.newaxis]
            # >> this one works for tt >> I need to check why...
        
        return arr_norm

        # normalize P values (between 0 and 1) so that they can reflect probabilities 
        # the probability with itself will be the highest
        # !!! IMPORTANT !!!
        # the probability with movie nodes will be ignored since we are looking for the relations(similarity-measure) 
        # with other users
        #P_norm = np.round(Normalize_Array(P), 6)
        
    def Generate_P_Matrix(self, max_connected_gr_amazon_movies, bottom_nodes, top_nodes):
        start_time = datetime.datetime.now()
        print('P to_numpy_matrix: {}'.format(datetime.datetime.now()))
        P = nx.to_numpy_matrix(max_connected_gr_amazon_movies)
        # by doing the below conversion, we are just considering the links between the nodes, we ignore the weights
        # we try to understand if considering the weights will benefit us or not
        #P[P > 0] = 1
        print('P normalize: {}'.format(datetime.datetime.now()))
        #P_norm = self.Normalize_Array(P)
        # below method is faster 
        P_norm = self.Normalize_Matrix(P, dim = 'row')

        n_nodes = P_norm.shape[0]
        # initiate R matrix which includes the r values of each node and
        # having the highest relation of the node by itself, setting the diagonal to 1
        R = np.diag(np.ones(n_nodes), 0)
        beta = 0.15
        #R *= beta
        R_zero = R.copy()
        #print(P)
        #print(P_norm)
        #print(R)
        #print(R_zero)
        end_time = datetime.datetime.now()

        print('P finalize: {}'.format(datetime.datetime.now()))
        print('Calculation time-P matrix generation from the graph: {}'.format(end_time-start_time))

        return P_norm, R, R_zero
    
    def Generate_P_Matrix_only(self, max_connected_gr_amazon_movies, bottom_nodes, top_nodes):
        start_time = datetime.datetime.now()
        print('P to_numpy_matrix: {}'.format(datetime.datetime.now()))
        P = nx.to_numpy_matrix(max_connected_gr_amazon_movies)
        # by doing the below conversion, we are just considering the links between the nodes, we ignore the weights
        # we try to understand if considering the weights will benefit us or not
        #P[P > 0] = 1
        print('P normalize: {}'.format(datetime.datetime.now()))
        #P_norm = self.Normalize_Array(P)
        # below method is faster 
        P_norm = self.Normalize_Matrix(P, dim = 'row')

        end_time = datetime.datetime.now()

        print('P finalize: {}'.format(datetime.datetime.now()))
        print('Calculation time-P matrix generation from the graph: {}'.format(end_time-start_time))

        return P_norm
    
    def generate_node_lists(self, max_connected_gr_amazon_movies, bottom_nodes, top_nodes):
        # TEST - see if the matrix-P shows the correct values (to be sure if it's not changing the order)
        # !!! IMPORTANT !!!
        # apparently, it doesn't write the movies first and users last or doesn't follow any order..
        '''
        node_0 = list(max_connected_gr_amazon_movies.nodes)[0]
        node_1 = list(max_connected_gr_amazon_movies.nodes)[1]
        edge_01 = max_connected_gr_amazon_movies.has_edge(node_0, node_1)
        if edge_01 == True:
            weight_01 = max_connected_gr_amazon_movies.get_edge_data(node_0, node_1)['weight']

        # Bxxxxxx = top_nodes = movies
        print('node0={} .. node1={} .. edge01={} .. weight01={}'.format(node_0, node_1, edge_01, weight_01))
        '''
        
        node_list = list(max_connected_gr_amazon_movies.nodes)
        bottom_node_list = list(bottom_nodes)
        top_node_list = list(top_nodes)

        self.n_reviews = len(max_connected_gr_amazon_movies.edges)
        self.n_users = len(bottom_nodes)
        self.n_movies = len(top_nodes)
        #print(node_list)
        #print('\nbottom_nodes={}'.format(bottom_node_list))
        #print('\ntop_nodes={}'.format(top_node_list))

        return node_list, bottom_node_list, top_node_list, self.n_reviews, self.n_users, self.n_movies
    
    def node_check_f(self, usr_idx = 0):
        # Bxxxxxx = top_nodes = movies
        node_check = node_list[usr_idx]
        if node_check in top_node_list:
            print('this is a movie: {}'.format(node_check))
        else:
            print('this is a user: {}'.format(node_check))

        #list(max_connected_gr_amazon_movies.edges([node_list[0]]))
        #max_connected_gr_amazon_movies.edges(['B003AI2VGA']) 
        #nx.edges(max_connected_gr_amazon_movies, ['B003AI2VGA'])
        #node_list.index('B003AI2VGA')
        
    def movie_user_indexes(self, node_list, top_node_list):
        start_time = datetime.datetime.now()
        user_indexes = []
        movie_indexes = []
        for nn in node_list:
            if nn in top_node_list:
                movie_indexes.append(node_list.index(nn))
            else:
                user_indexes.append(node_list.index(nn))

        end_time = datetime.datetime.now()
        print('Calculation time-user-movie indexes: {}'.format(end_time-start_time))
        print('\n')

        if self.debug_mode == 'On':
            print('\n15 movie indexes just to give some samples:')
            print(movie_indexes[0:15])
            # users are usually too many, so will not print them
            #print(user_indexes)

        return movie_indexes, user_indexes
    
    def random_walk_vector(self, P_norm, R, R_zero, beta=0.15, n_steps=3):
        # !!!!
        # here, we will pass a vector of the interested user from the R matrix, we will not pass the entire matrix
        # it will be like:
        # R[user_index]  and R_zero[user_index]

        # with 40K records selected, 
        # MEMORY of python hits to 45GB at some point and works around 34GB in general
        # CPU works at around 390%
        # there should be a better way to do this...
        # tried cupy but it fails in Mac (latest OS version doesn't have support)
        # Colab shows like importing cupy but then during the execution it fails at some point, couldn't fix it... 
        # need to retry in a Windows laptop if cupy works or not..
        # but even if it works, the number users reviewing more than 1 movie is too few

        #### Instead of calculating it for all the nodes, maybe we should calculate it node by node each time
        #### meaning that, instead of using a R-matrix, we can use a r-vector only. That is also how it is done
        #### in the notes as I remember...
        start_time = datetime.datetime.now()
        #n_steps = 2

        for i in range(n_steps):
            R = (1-beta)*np.dot(R, P_norm) + beta*R_zero
            if self.debug_mode == 'On':
                print('step-{} completed .. date: {}'.format(i, datetime.datetime.now()))

        #print(R)
        end_time = datetime.datetime.now()
        if self.debug_mode == 'On':
            print('Calculation time-steps walked: {}'.format(end_time-start_time))
        return R

    #def similarity_check_vector(R, ref_user_idx, top_similarity=40):
    def similarity_check_vector(self, R, max_connected_gr_amazon_movies, movie_indexes, ref_user_idx, node_list, top_similarity=40):
        # !!!!
        # here, we will pass a vector of the interested user from the R matrix, we will not pass the entire matrix
        # it will be like:
        # R[user_index]  and R_zero[user_index]

        # user similarity check
        # fetch the values for node=0 (which is a user)
        #ref_user_idx = 3 #23746
        # we will pick the most similar 10 users in this case
        # this value should be picked as around the below ratio and I believe, 
        # if any of the top-similar users reviewed a movie, 
        # we can guess like our user will watch the movie: WILL BE TESTED
        # OR we can pick a high number (>ratio) for the top similarity and the percentage>10 maybe counted as the candidate
        #top_similarity = n_reviews/n_movies
        #top_similarity = 40
        user_test = np.array(list(R))
        if self.debug_mode == 'On':
            print('R matrix values for the user (r vector of the user): \n{}'.format(user_test))

        # sort the values of each column from smaller to the bigger
        sorted_nodes = np.argsort(user_test)

        user_similarity = []

        # we will not consider the user himself or movies as similar nodes
        for idx in sorted_nodes[0][0]:
            if idx != ref_user_idx and idx not in movie_indexes:
                user_similarity.append(idx)

        user_similarity_top = user_similarity[-top_similarity:]

        # ALL is too many, so we will not print it
        #print('Ordered similarities-ALL: {}'.format(user_similarity))
        if self.debug_mode == 'On':
            print('\n Top similarities: {}'.format(user_similarity_top))

        #for usr in user_similarity_top:
        #    print(list(max_connected_gr_amazon_movies.nodes)[user_similarity_top[usr]])
        #print(list(max_connected_gr_amazon_movies.nodes)[user_similarity_top[0]])

        user_similarity_top_non_zero = []
        user_similarity_top_score = []
        # in some cases, we recognized that some of the top neighbors have 0 probabilities
        # when we remove those neighbor, the results look better
        user_similarity_top_score_non_zero = []
        for usr_top in user_similarity_top:
            scr_top = round(R[0, usr_top], 4)
            user_similarity_top_score.append(scr_top)
            if scr_top > 0:
                user_similarity_top_non_zero.append(usr_top)
                user_similarity_top_score_non_zero.append(scr_top)

        if self.debug_mode == 'On':
            print('\n user_similarity_top_score: {}'.format(user_similarity_top_score))
            print('\n user_similarity_top_non_zero: {}'.format(user_similarity_top_non_zero))
            print('\n user_similarity_top_score_non_zero: {}'.format(user_similarity_top_score_non_zero))

        # top_nodes = MOVIES, bottom_nodes = users
        user_reviewed = []
        user_reviewed_id = []
        movie_list_ref = []
        movie_reviewed = []
        movie_reviewed_id = []
        is_reviewed = []
        movie_score = []
        review_count_list = []

        movie_by_reference_user = list(max_connected_gr_amazon_movies.edges([node_list[ref_user_idx]]))

        for edge_m in movie_by_reference_user:
            # if the second item is a user then we will add the first item to the movie_list
            if edge_m[1].startswith('A'):
                movie_list_ref.append(edge_m[0])
            else:
                movie_list_ref.append(edge_m[1])

        for u_idx in user_similarity_top_non_zero:
            user_name = node_list[u_idx]
            for m in movie_list_ref:
                m_idx = node_list.index(m)
                is_reviewed.append(max_connected_gr_amazon_movies.has_edge(user_name, m))
                user_reviewed.append(user_name)
                user_reviewed_id.append(u_idx)
                movie_reviewed.append(m)
                movie_reviewed_id.append(m_idx)
                try:
                    scr = max_connected_gr_amazon_movies.get_edge_data(user_name, m)['weight']
                except:
                    scr = '...'  # NA - no review score
                #scr = str(max_connected_gr_amazon_movies.get_edge_data(node_list[u_idx], m)['weight'])
                #max_connected_gr_amazon_movies.get_edge_data(node_list[u_idx], 'B003AI2VGA')['weight']
                movie_score.append(scr)
                review_count = len(list(max_connected_gr_amazon_movies.edges([m])))
                review_count_list.append(review_count)

                #is_reviewed = max_connected_gr_amazon_movies.get_edge_data(u, m)['weight']


        dict_summary = {'user': user_reviewed, 'movie': movie_reviewed, 
                        'user_id': user_reviewed_id, 'movie_id': movie_reviewed_id, 
                        'is_reviewed': is_reviewed, 'score': movie_score,
                       'n_review': review_count_list}

        '''
        dict_summary = {'user': user_reviewed, 'movie': movie_reviewed, 
                        'user_id': user_reviewed_id, 'movie_id': movie_reviewed_id, 
                        'is_reviewed': is_reviewed}
        '''

        print('\n')

        df_summary = pd.DataFrame(dict_summary)

        print('# of users in the evaluation: {}'.format(top_similarity))

        '''
        true_values=len(df_summary.query("movie=='B003AI2VGA' and is_reviewed==True"))
        total_values=len(df_summary.query("movie=='B003AI2VGA'"))

        # review ratio by similar users
        ratio_similar = true_values/total_values*100
        print(ratio_similar)
        '''

        # DOUBLE CHECK if edge calculation is correct
        # The problem is many of the users just review 1 movie or very few movies
        # total review count per user. How many movies did a specific user reviewed?

        '''
        user_review_counts = []
        user_review_counts_ALL = 0
        # total review count per movie. How many users did a specific movie reviewed by?
        movie_review_counts = []
        movie_review_counts_ALL = 0

        for usr in bottom_node_list:
            usr_count = len(list(max_connected_gr_amazon_movies.edges([usr])))
            user_review_counts.append([usr, usr_count])
            user_review_counts_ALL += usr_count

        for mov in top_node_list:
            mov_count = len(list(max_connected_gr_amazon_movies.edges([mov])))
            user_review_counts.append([mov, mov_count])
            movie_review_counts_ALL += mov_count

        print(user_review_counts)
        '''

        return df_summary, movie_list_ref, movie_by_reference_user, user_similarity_top_non_zero
    
    def similarity_check_vector_VAL(self, R, movie_indexes, ref_user_idx, node_list, node_list_v, movie_FOR_reference_user, max_connected_gr_amazon_movies, max_connected_gr_amazon_movies_v, top_similarity=40):
        # !!!!
        # here, we will pass a vector of the interested user from the R matrix, we will not pass the entire matrix
        # it will be like:
        # R[user_index]  and R_zero[user_index]

        # user similarity check
        # fetch the values for node=0 (which is a user)
        #ref_user_idx = 3 #23746
        # we will pick the most similar 10 users in this case
        # this value should be picked as around the below ratio and I believe, 
        # if any of the top-similar users reviewed a movie, 
        # we can guess like our user will watch the movie: WILL BE TESTED
        # OR we can pick a high number (>ratio) for the top similarity and the percentage>10 maybe counted as the candidate
        #top_similarity = n_reviews/n_movies
        #top_similarity = 40
        user_test = np.array(list(R))
        if self.debug_mode == 'On':
            print('R matrix values for the user (r vector of the user): \n{}'.format(user_test))

        # sort the values of each column from smaller to the bigger
        sorted_nodes = np.argsort(user_test)

        user_similarity = []

        # we will not consider the user himself or movies as similar nodes
        for idx in sorted_nodes[0][0]:
            if idx != ref_user_idx and idx not in movie_indexes:
                user_similarity.append(idx)

        user_similarity_top = user_similarity[-top_similarity:]

        # ALL is too many, so we will not print it
        #print('Ordered similarities-ALL: {}'.format(user_similarity))
        if self.debug_mode == 'On':
            print('\n Top similarities: {}'.format(user_similarity_top))

        #for usr in user_similarity_top:
        #    print(list(max_connected_gr_amazon_movies.nodes)[user_similarity_top[usr]])
        #print(list(max_connected_gr_amazon_movies.nodes)[user_similarity_top[0]])

        user_similarity_top_non_zero = []
        user_similarity_top_score = []
        # in some cases, we recognized that some of the top neighbors have 0 probabilities
        # when we remove those neighbor, the results look better
        user_similarity_top_score_non_zero = []
        for usr_top in user_similarity_top:
            scr_top = round(R[0, usr_top], 4)
            user_similarity_top_score.append(scr_top)
            if scr_top > 0:
                user_similarity_top_non_zero.append(usr_top)
                user_similarity_top_score_non_zero.append(scr_top)

        if self.debug_mode == 'On':
            print('\n user_similarity_top_score: {}'.format(user_similarity_top_score))
            print('\n user_similarity_top_non_zero: {}'.format(user_similarity_top_non_zero))
            print('\n user_similarity_top_score_non_zero: {}'.format(user_similarity_top_score_non_zero))

        # top_nodes = MOVIES, bottom_nodes = users
        # !!!!!
        # now, only checking for the old users if they created a new edge or not
        # the check can be extended for the new users (can be? if the user is new, there will be no neighbors
        # from the random walk in the previous dataset. if we make a random walk in the new dataset, then we will
        # already know the result for the user and it would be including in the calculations. on the other hand,
        # if we think of removing the user form the second dataset, then the second dataset will lose its
        # bipartite )
        user_reviewed = []
        user_reviewed_id = []
        movie_list_ref = []
        movie_reviewed = []
        movie_reviewed_id = []
        is_reviewed = []
        movie_score = []
        review_count_list = []

        #movie_by_reference_user = list(max_connected_gr_amazon_movies.edges([node_list[ref_user_idx]]))

        ''''
        # here we don't pass edges, we directly pass the whole movies in a list format
        for edge_m in movie_FOR_reference_user:
            # if the second item is a user then we will add the first item to the movie_list
            if edge_m[1].startswith('A'):
                movie_list_ref.append(edge_m[0])
            else:
                movie_list_ref.append(edge_m[1])
        '''
        movie_list_ref = movie_FOR_reference_user

        is_reviewed_old = False
        is_reviewed_new = False
        for u_idx in user_similarity_top_non_zero:
            user_name = node_list[u_idx]
                          
            for m in movie_list_ref:
                #m_idx = node_list.index(m)
                # we are looking for the edges in the new dataset so we will use the movies in the new one
                # the same movie can exist in the first dataset but with different edges
                # meaning that, in the first dataset, the user might not have an edge with this movie
                # but he can make a review at a later time (t+1)
                # that's why the search is done in the new one and index is from the new dataset
                m_idx = node_list_v.index(m)

                # some movies will be new and they will not exist in the old dataset
                try:
                    is_reviewed_old = max_connected_gr_amazon_movies.has_edge(user_name, m)
                except:
                    is_reviewed_old = False

                # the user might not be in the second dataset, so it can throw an exception
                # if the user is not there, then is_reviewed_new = False
                try:
                    is_reviewed_new = max_connected_gr_amazon_movies_v.has_edge(user_name, m)
                except:
                    is_reviewed_new = False

                # if there is an edge in any of the datasets, we will set the value as TRUE
                is_reviewed.append(is_reviewed_old or is_reviewed_new)
                user_reviewed.append(user_name)
                user_reviewed_id.append(u_idx)
                movie_reviewed.append(m)
                movie_reviewed_id.append(m_idx)

                scr_old = '...'
                scr_new = '...'
                try:
                    scr_old = max_connected_gr_amazon_movies.get_edge_data(user_name, m)['weight']
                except:
                    scr_old = '...'  # NA - no review score

                try:
                    scr_new = max_connected_gr_amazon_movies_v.get_edge_data(user_name, m)['weight']
                except:
                    scr_new = '...'  # NA - no review score

                if scr_new != '...':
                    scr = scr_new
                else:
                    scr = scr_old

                #scr = str(max_connected_gr_amazon_movies.get_edge_data(node_list[u_idx], m)['weight'])
                #max_connected_gr_amazon_movies.get_edge_data(node_list[u_idx], 'B003AI2VGA')['weight']
                movie_score.append(scr)

                review_count_old = 0
                review_count_new = 0

                try:
                    review_count_old = len(list(max_connected_gr_amazon_movies.edges([m])))
                except:
                    review_count_old = 0

                try:
                    review_count_new = len(list(max_connected_gr_amazon_movies_v.edges([m])))
                except:
                    review_count_new = 0

                review_count = review_count_old + review_count_new
                review_count_list.append(review_count)

                #is_reviewed = max_connected_gr_amazon_movies.get_edge_data(u, m)['weight']


        dict_summary = {'user': user_reviewed, 'movie': movie_reviewed, 
                        'user_id': user_reviewed_id, 'movie_id': movie_reviewed_id, 
                        'is_reviewed': is_reviewed, 'score': movie_score,
                       'n_review': review_count_list}

        '''
        dict_summary = {'user': user_reviewed, 'movie': movie_reviewed, 
                        'user_id': user_reviewed_id, 'movie_id': movie_reviewed_id, 
                        'is_reviewed': is_reviewed}
        '''

        #print('\n')

        df_summary = pd.DataFrame(dict_summary)

        if self.debug_mode == 'On':
            print('# of users in the evaluation: {}'.format(top_similarity))

        '''
        true_values=len(df_summary.query("movie=='B003AI2VGA' and is_reviewed==True"))
        total_values=len(df_summary.query("movie=='B003AI2VGA'"))

        # review ratio by similar users
        ratio_similar = true_values/total_values*100
        print(ratio_similar)
        '''

        # DOUBLE CHECK if edge calculation is correct
        # The problem is many of the users just review 1 movie or very few movies
        # total review count per user. How many movies did a specific user reviewed?

        '''
        user_review_counts = []
        user_review_counts_ALL = 0
        # total review count per movie. How many users did a specific movie reviewed by?
        movie_review_counts = []
        movie_review_counts_ALL = 0

        for usr in bottom_node_list:
            usr_count = len(list(max_connected_gr_amazon_movies.edges([usr])))
            user_review_counts.append([usr, usr_count])
            user_review_counts_ALL += usr_count

        for mov in top_node_list:
            mov_count = len(list(max_connected_gr_amazon_movies.edges([mov])))
            user_review_counts.append([mov, mov_count])
            movie_review_counts_ALL += mov_count

        print(user_review_counts)
        '''

        return df_summary, movie_list_ref, movie_FOR_reference_user, user_similarity_top_non_zero
    
    def similarity_check_vector_VAL_model3(self, R, movie_indexes, ref_user_idx, node_list, node_list_v, movie_FOR_reference_user, max_connected_gr_amazon_movies, max_connected_gr_amazon_movies_v, top_similarity=40):
        # !!!!
        # here, we will pass a vector of the interested user from the R matrix, we will not pass the entire matrix
        # it will be like:
        # R[user_index]  and R_zero[user_index]

        # user similarity check
        # fetch the values for node=0 (which is a user)
        #ref_user_idx = 3 #23746
        # we will pick the most similar 10 users in this case
        # this value should be picked as around the below ratio and I believe, 
        # if any of the top-similar users reviewed a movie, 
        # we can guess like our user will watch the movie: WILL BE TESTED
        # OR we can pick a high number (>ratio) for the top similarity and the percentage>10 maybe counted as the candidate
        #top_similarity = n_reviews/n_movies
        #top_similarity = 40
        user_test = np.array(list(R))
        if self.debug_mode == 'On':
            print('R matrix values for the user (r vector of the user): \n{}'.format(user_test))

        # sort the values of each column from smaller to the bigger
        sorted_nodes = np.argsort(user_test)

        user_similarity = []

        # we will not consider the user himself or movies as similar nodes
        for idx in sorted_nodes[0][0]:
            if idx != ref_user_idx and idx not in movie_indexes:
                user_similarity.append(idx)

        user_similarity_top = user_similarity[-top_similarity:]

        # ALL is too many, so we will not print it
        #print('Ordered similarities-ALL: {}'.format(user_similarity))
        if self.debug_mode == 'On':
            print('\n Top similarities: {}'.format(user_similarity_top))

        #for usr in user_similarity_top:
        #    print(list(max_connected_gr_amazon_movies.nodes)[user_similarity_top[usr]])
        #print(list(max_connected_gr_amazon_movies.nodes)[user_similarity_top[0]])

        user_similarity_top_non_zero = []
        user_similarity_top_score = []
        # in some cases, we recognized that some of the top neighbors have 0 probabilities
        # when we remove those neighbor, the results look better
        user_similarity_top_score_non_zero = []
        for usr_top in user_similarity_top:
            scr_top = round(R[0, usr_top], 4)
            user_similarity_top_score.append(scr_top)
            if scr_top > 0:
                user_similarity_top_non_zero.append(usr_top)
                user_similarity_top_score_non_zero.append(scr_top)

        if self.debug_mode == 'On':
            print('\n user_similarity_top_score: {}'.format(user_similarity_top_score))
            print('\n user_similarity_top_non_zero: {}'.format(user_similarity_top_non_zero))
            print('\n user_similarity_top_score_non_zero: {}'.format(user_similarity_top_score_non_zero))

        # top_nodes = MOVIES, bottom_nodes = users
        # !!!!!
        # now, only checking for the old users if they created a new edge or not
        # the check can be extended for the new users (can be? if the user is new, there will be no neighbors
        # from the random walk in the previous dataset. if we make a random walk in the new dataset, then we will
        # already know the result for the user and it would be including in the calculations. on the other hand,
        # if we think of removing the user form the second dataset, then the second dataset will lose its
        # bipartite )
        user_reviewed = []
        user_reviewed_id = []
        movie_list_ref = []
        movie_reviewed = []
        movie_reviewed_id = []
        is_reviewed = []
        movie_score = []
        review_count_list = []

        #movie_by_reference_user = list(max_connected_gr_amazon_movies.edges([node_list[ref_user_idx]]))

        ''''
        # here we don't pass edges, we directly pass the whole movies in a list format
        for edge_m in movie_FOR_reference_user:
            # if the second item is a user then we will add the first item to the movie_list
            if edge_m[1].startswith('A'):
                movie_list_ref.append(edge_m[0])
            else:
                movie_list_ref.append(edge_m[1])
        '''
        movie_list_ref = movie_FOR_reference_user

        is_reviewed_old = False
        is_reviewed_new = False
        for u_idx in user_similarity_top_non_zero:
            user_name = node_list[u_idx]
            for m in movie_list_ref:
                #m_idx = node_list.index(m)
                # we are looking for the edges in the new dataset so we will use the movies in the new one
                # the same movie can exist in the first dataset but with different edges
                # meaning that, in the first dataset, the user might not have an edge with this movie
                # but he can make a review at a later time (t+1)
                # that's why the search is done in the new one and index is from the new dataset
                m_idx = node_list_v.index(m)

                # some movies will be new and they will not exist in the old dataset
                try:
                    is_reviewed_old = max_connected_gr_amazon_movies.has_edge(user_name, m)
                except:
                    is_reviewed_old = False

                # the user might not be in the second dataset, so it can throw an exception
                # if the user is not there, then is_reviewed_new = False
                try:
                    is_reviewed_new = max_connected_gr_amazon_movies_v.has_edge(user_name, m)
                except:
                    is_reviewed_new = False

                # if there is an edge in any of the datasets, we will set the value as TRUE
                is_reviewed.append(is_reviewed_old or is_reviewed_new)
                user_reviewed.append(user_name)
                user_reviewed_id.append(u_idx)
                movie_reviewed.append(m)
                movie_reviewed_id.append(m_idx)

                scr_old = '...'
                scr_new = '...'
                try:
                    scr_old = max_connected_gr_amazon_movies.get_edge_data(user_name, m)['weight']
                except:
                    scr_old = '...'  # NA - no review score

                try:
                    scr_new = max_connected_gr_amazon_movies_v.get_edge_data(user_name, m)['weight']
                except:
                    scr_new = '...'  # NA - no review score

                if scr_new != '...':
                    scr = scr_new
                else:
                    scr = scr_old

                #scr = str(max_connected_gr_amazon_movies.get_edge_data(node_list[u_idx], m)['weight'])
                #max_connected_gr_amazon_movies.get_edge_data(node_list[u_idx], 'B003AI2VGA')['weight']
                movie_score.append(scr)

                review_count_old = 0
                review_count_new = 0

                try:
                    review_count_old = len(list(max_connected_gr_amazon_movies.edges([m])))
                except:
                    review_count_old = 0

                try:
                    review_count_new = len(list(max_connected_gr_amazon_movies_v.edges([m])))
                except:
                    review_count_new = 0

                review_count = review_count_old + review_count_new
                review_count_list.append(review_count)

                #is_reviewed = max_connected_gr_amazon_movies.get_edge_data(u, m)['weight']


        dict_summary = {'user': user_reviewed, 'movie': movie_reviewed, 
                        'user_id': user_reviewed_id, 'movie_id': movie_reviewed_id, 
                        'is_reviewed': is_reviewed, 'score': movie_score,
                       'n_review': review_count_list}

        '''
        dict_summary = {'user': user_reviewed, 'movie': movie_reviewed, 
                        'user_id': user_reviewed_id, 'movie_id': movie_reviewed_id, 
                        'is_reviewed': is_reviewed}
        '''

        #print('\n')

        df_summary = pd.DataFrame(dict_summary)

        if self.debug_mode == 'On':
            print('# of users in the evaluation: {}'.format(top_similarity))

        '''
        true_values=len(df_summary.query("movie=='B003AI2VGA' and is_reviewed==True"))
        total_values=len(df_summary.query("movie=='B003AI2VGA'"))

        # review ratio by similar users
        ratio_similar = true_values/total_values*100
        print(ratio_similar)
        '''

        # DOUBLE CHECK if edge calculation is correct
        # The problem is many of the users just review 1 movie or very few movies
        # total review count per user. How many movies did a specific user reviewed?

        '''
        user_review_counts = []
        user_review_counts_ALL = 0
        # total review count per movie. How many users did a specific movie reviewed by?
        movie_review_counts = []
        movie_review_counts_ALL = 0

        for usr in bottom_node_list:
            usr_count = len(list(max_connected_gr_amazon_movies.edges([usr])))
            user_review_counts.append([usr, usr_count])
            user_review_counts_ALL += usr_count

        for mov in top_node_list:
            mov_count = len(list(max_connected_gr_amazon_movies.edges([mov])))
            user_review_counts.append([mov, mov_count])
            movie_review_counts_ALL += mov_count

        print(user_review_counts)
        '''

        return df_summary, movie_list_ref, movie_FOR_reference_user, user_similarity_top_non_zero
    
    def similarity_summary_ratio(self, df_summary, movie_list_ref, max_connected_gr_amazon_movies, movie_by_reference_user):
        ratio_review_list = []
        review_count_list2 = []

        for m in movie_list_ref:
            true_values = len(df_summary.query("movie=='" + m + "' and is_reviewed==True"))
            total_values = len(df_summary.query("movie=='" + m + "'"))
            review_count = len(list(max_connected_gr_amazon_movies.edges([m])))
            # review ratio by similar users
            ratio_similar = true_values/total_values*100
            ratio_review_list.append(ratio_similar)
            review_count_list2.append(review_count)

        #dict_summary_ratio = {'movie': movie_list_ref, 'ratio_similar': ratio_review_list}
        dict_summary_ratio = {'user_movie': movie_by_reference_user, 'ratio_similar': ratio_review_list,
                             'review_count': review_count_list2}

        df_summary_ratio = pd.DataFrame(dict_summary_ratio)
        #print('# of users in the evaluation: {}'.format(top_similarity))
        if self.debug_mode == 'On':
            print('Total # of MOVIEs: {} USERS: {} REVIEWs: {},  in the graph'.format(self.n_movies, self.n_users, self.n_reviews))
            print('\n')
            print(df_summary_ratio)

        return df_summary_ratio
    
    def similarity_summary_ratio_VAL(self, df_summary, movie_list_ref, max_connected_gr_amazon_movies, max_connected_gr_amazon_movies_v):
        ratio_review_list = []
        review_count_list2 = []
        review_count_list2_OLD = []
        review_count_list2_NEW = []

        for m in movie_list_ref:
            true_values = len(df_summary.query("movie=='" + m + "' and is_reviewed==True"))
            total_values = len(df_summary.query("movie=='" + m + "'"))
            review_count_old = len(list(max_connected_gr_amazon_movies.edges([m])))
            review_count_new = len(list(max_connected_gr_amazon_movies_v.edges([m])))
            review_count = review_count_old + review_count_new
            # review ratio by similar users
            ratio_similar = true_values/total_values*100
            ratio_review_list.append(ratio_similar)
            review_count_list2.append(review_count)
            review_count_list2_OLD.append(review_count_old)
            review_count_list2_NEW.append(review_count_new)

        #dict_summary_ratio = {'movie': movie_list_ref, 'ratio_similar': ratio_review_list}
        dict_summary_ratio = {'movie': movie_list_ref, 'ratio_similar': ratio_review_list,
                             'review_count': review_count_list2, 'review_count_old': review_count_list2_OLD, 'review_count_new': review_count_list2_NEW}

        df_summary_ratio = pd.DataFrame(dict_summary_ratio)
        #print('# of users in the evaluation: {}'.format(top_similarity))
        if self.debug_mode == 'On':
            print('Total # of MOVIEs: {} USERS: {} REVIEWs: {},  in the graph'.format(self.n_movies, self.n_users, self.n_reviews))
            print('\n')
            print(df_summary_ratio)

        return df_summary_ratio

    def movie_user_compare_datasets(self, top_nodes_v, top_nodes, bottom_nodes_v, bottom_nodes):
        count_YES = 0
        count_NO = 0
        for mv in top_nodes_v:
            if mv in top_nodes:
                #print('mv={} .. {}'.format(mv, 'YES'))
                count_YES += 1
            else:
                #print('mv={} .. {}'.format(mv, 'NO'))
                count_NO += 1

        # all movies in dataset2 exist in dataset1
        print('MOVIES >> YES: Exist in both .. NO: Exist in dataset-2 but not exist in dataset-1')
        print('count_YES: {} .. count_NO: {}'.format(count_YES, count_NO))

        count_YES = 0
        count_NO = 0
        for usr in bottom_nodes_v:
            if usr in bottom_nodes:
                #print('mv={} .. {}'.format(mv, 'YES'))
                count_YES += 1
            else:
                #print('mv={} .. {}'.format(mv, 'NO'))
                count_NO += 1

        # also all users in dataset2 exist in dataset1
        print('USERS >> YES: Exist in both .. NO: Exist in dataset-2 but not exist in dataset-1')
        print('count_YES: {} .. count_NO: {}'.format(count_YES, count_NO))

        count_YES = 0
        count_NO = 0
        for mv in top_nodes:
            if mv in top_nodes_v:
                #print('mv={} .. {}'.format(mv, 'YES'))
                count_YES += 1
            else:
                #print('mv={} .. {}'.format(mv, 'NO'))
                count_NO += 1

        # dataset1 has 694 more movies than the dataset2
        print('MOVIES >> YES: Exist in both .. NO: Exist in dataset-1 but not exist in dataset-2')
        print('count_YES: {} .. count_NO: {}'.format(count_YES, count_NO))

        count_YES = 0
        count_NO = 0
        for usr in bottom_nodes:
            if usr in bottom_nodes_v:
                #print('mv={} .. {}'.format(mv, 'YES'))
                count_YES += 1
            else:
                #print('mv={} .. {}'.format(mv, 'NO'))
                count_NO += 1

        # dataset1 has 25254 more users than the dataset2
        print('USERS >> YES: Exist in both .. NO: Exist in dataset-1 but not exist in dataset-2')
        print('count_YES: {} .. count_NO: {}'.format(count_YES, count_NO))
        