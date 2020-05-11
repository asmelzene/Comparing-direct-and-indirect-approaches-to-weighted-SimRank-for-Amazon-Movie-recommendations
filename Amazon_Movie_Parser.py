import fileinput
import numpy as np
import datetime
import pandas as pd

def AmazonMovies(num_reviews, file_name="movies.txt", outputTo='screen'):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    *3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    6) review/time: 1182729600
    7) review/summary: "There Is So Much Darkness Now ~ Come For The Miracle"
    8) review/text: Synopsis: On the daily..............
    9) null
    '''
    i = 0
    single_entry=np.zeros(4, str)
    single_entry_completed = True

    movie_data = np.array([["product/productId", "review/userId", "review/profileName", "review/score"]])
    #movie_data = np.array([["product/productId", "review/userId", "review/profileName", \
     #                       "review/helpfulness", "review/score"]])
    num_rec = 0

    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            i += 1
            #for each entry, repeats at every 9
            if (i % 9 == 1):
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                single_entry=np.array([["productId", "userId", "profileName", "score"]])
                #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                single_entry[0, 0] = line[19:].rstrip("\n")
            elif (i % 9 == 2):
                single_entry[0, 1] = line[15:].rstrip("\n")
            elif (i % 9 == 3):
                single_entry[0, 2] = line[20:].rstrip("\n")
            elif (i % 9 == 5):
                single_entry[0, 3] = line[14:].rstrip("\n")
            elif (i % 9 == 0):
                num_rec += 1
                #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                movie_data = np.append(movie_data, single_entry, axis = 0)

            if (num_rec == num_reviews):
                fileinput.close()
                break

    if outputTo == 'file':
        # to avoid overwriting an existing file, if you want to overwrite, just basically use output.txt
        file_name = FileNameUnique() 
        #with open("output.txt", "w") as output:
        with open(file_name, "w") as output:
            for movie in movie_data:
                output.write(str(movie) + "\n")
    elif outputTo == 'screen':
        #if you just want to show it here, uncomment to above part
        for movie in movie_data:
            print(str(movie) + "\n")
    elif outputTo == 'dictionary':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        return movie_dict
def Time_Stamp():
    date_time = datetime.datetime.now()
    
    D = str(date_time.day)
    M = str(date_time.month)
    Y = str(date_time.year)

    h = str(date_time.hour)
    m = str(date_time.minute)
    s = str(date_time.second)
    
    date_array = [D, M, Y, h, m, s]
    
    return date_array
    
def FileNameUnique(prefix = "Movies_", suffix = '.txt'):
    file_name = prefix

    date_array = Time_Stamp()
    
    for idx, i in enumerate(date_array):
        if idx == 2:
            file_name += i + '_'
        elif idx == 5:
            file_name += i + suffix
        else:
            file_name += i + '.'

    #print(file_name)
    return file_name
                
def Save_Movies(movie_graph, prefix = "Movies_Connected_"):
    # ('userID', 'movieID', 'score')
    # ('A141HP4LYPW', 'B003AI2VGA', '3.0')
    movie_list = list(movie_graph.edges.data('weight', default=1))
    
    file_name = prefix + str(len(movie_list)) + "_"
    date_array = Time_Stamp()
    
    for idx, i in enumerate(date_array):
        if idx == 2:
            file_name += i + '_'  # YEAR
        elif idx == 5:
            file_name += i + '.txt'  # SECOND
        else:
            file_name += i + '.'
 
    # since the graph is undirected, once the edges is created as (u, v, weight)
    # it was observed that some edges are saved in the graph as (v, u, weight) # graph.edges returns some edges in that way
    # so, for the second use, to make the life easier for us, we save them by checking if they are in the right order
    with open(file_name, "w") as output:
        for movie in movie_list:
            if movie[0].startswith('A'):
                userId = movie[0]
                movieId = movie[1]
            else:
                #elif movie[0].startswith('B'):
                # some MOVIES don't start with B but instead full numbers like: 0790747324
                userId = movie[1]
                movieId = movie[0]
            str_movie = userId + "," + movieId + "," + movie[2]
            #output.write(str(movie) + "\n")
            output.write(str_movie + "\n")
    print('Graph info saved in: {}'.format(file_name))
    return file_name
    
def Reorganize_Edges_Graph(movie_graph):
    # ('userID', 'movieID', 'score')
    # ('A141HP4LYPW', 'B003AI2VGA', '3.0')
    movie_list = list(movie_graph.edges.data('weight', default=1))
    movie_list_organized = []

    # since the graph is undirected, once the edges is created as (u, v, weight)
    # it was observed that some edges are saved in the graph as (v, u, weight) # graph.edges returns some edges in that way
    # so, for the second use, to make the life easier for us, we re-organize them by checking if they are in the right order
    for movie in movie_list:
        if movie[1].startswith('A'):
            # if the second item is a user not a movie, we will switch their places
            userId = movie[0]
            movieId = movie[1]
        else:
            userId = movie[1]
            movieId = movie[0]

        organized_row = [userId, movieId, movie[2]]
        movie_list_organized.append(organized_row)
            
    print('Graph has been reorganized as a list, row format: [\'userID\', \'movieID\', \'score\']')
    return movie_list_organized

def Reorganize_Edges_List(movie_list):
    movie_list_organized = []

    # since the graph is undirected, once the edges is created as (u, v, weight)
    # it was observed that some edges are saved in the graph as (v, u, weight) # graph.edges returns some edges in that way
    # so, for the second use, to make the life easier for us, we re-organize them by checking if they are in the right order
    for movie in movie_list:
        if movie[1].startswith('A'):
            # if the second item is a user not a movie, we will switch their places
            userId = movie[0]
            movieId = movie[1]
        else:
            userId = movie[1]
            movieId = movie[0]

        organized_row = [userId, movieId, movie[2]]
        movie_list_organized.append(organized_row)
            
    print('Graph has been reorganized as a list, row format: [\'userID\', \'movieID\', \'score\']')
    return movie_list_organized
    
def Reorganize_Edges_DataFrame(movie_graph, input_type='List'):
    if input_type == 'Graph':
        movie_list = list(movie_graph.edges.data('weight', default=1))
    elif input_type == 'List':
        movie_list = movie_graph
    
    userId_list = []
    movieId_list = []
    score_list = []

    # since the graph is undirected, once the edges is created as (u, v, weight)
    # it was observed that some edges are saved in the graph as (v, u, weight) # graph.edges returns some edges in that way
    # so, for the second use, to make the life easier for us, we re-organize them by checking if they are in the right order
    for movie in movie_list:
        if movie[1].startswith('A'):
            # if the second item is a user not a movie, we will switch their places
            userId_list.append(movie[0])
            movieId_list.append(movie[1])
            score_list.append(movie[2])
        else:
            userId_list.append(movie[1])
            movieId_list.append(movie[0])
            score_list.append(movie[2])

        # Define a dictionary 
        dict_movies = {'userId':userId_list,
        'movieId':movieId_list,
        'score':score_list}
 
        # Convert the dictionary into DataFrame 
        df_movies = pd.DataFrame(dict_movies)
            
    print('Graph has been reorganized as a DataFrame, row format: [\'userId\', \'movieId\', \'score\']')
    return df_movies    
    
def Read_Connected_Movies_x(file_name):
    #movie_list = np.array([["userId", "productId", "score"]])
    movie_list = np.array([('userID', 'movieID', 'score')])
    print(movie_list[0])
    
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            print(line)
            movie_list = np.append(movie_list, line, axis = 0)
               
    return list(movie_list)

def Read_Connected_Movies(file_name):
    #movie_list = np.array([["userId", "productId", "score"]])
    movie_list = []
    
    with open(file_name, encoding="latin-1") as input_file:
        #movie_list = input_file.read().splitlines()
        #movie_list = input_file.readlines()
        for line in input_file:
            movie_list.append(line.rstrip("\n").split(sep=","))
               
    return movie_list