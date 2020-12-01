# Comparing direct and indirect approaches to weighted SimRank for Amazon Movie recommendations

the data source used:
https://snap.stanford.edu/data/web-Movies.html

the below file was unzipped and since it was too big to open, a parser was written to save a part of it after unzipping the file on the harddisk as movies.txt

movies.txt.gz	Amazon movie data (~8 million reviews)

product/productId: B00006HAXW
review/userId: A1RSDE90N6RSZF
review/profileName: Joseph M. Kotow
review/helpfulness: 9/9
review/score: 5.0
review/time: 1042502400
review/summary: Pittsburgh - Home of the OLDIES
review/text: I have all of the doo wop DVD's and this one is as good or better than the
1st ones. Remember once these performers are gone, we'll never get to see them again.
Rhino did an excellent job and if you like or love doo wop and Rock n Roll you'll LOVE
this DVD !!
-----------
save_objects_to_file and load_objects_from_file functions were added for objects
graph ("<class 'networkx.classes.graph.Graph'>")
matrix ("<class 'numpy.matrix'>")
dataframe ("<class 'pandas.core.frame.DataFrame'>")

in case we would like to play on the results at a later time, we can save the necessary information to files to avoid doing the same calculations again.

also, save_parameter_used function was added to save the parameter used. i.e.
parameter_list = ["file_name = 'data/movies.txt'", "n_movies = 40000", "n_movies_v = 5000", "ref_user_idx = 652",
                 "walk_steps = 40", "beta=0.15", "top_neighbor=40", "n_test_users = 100", "threshold=10"]
