import fileinput
import numpy as np
import datetime
import pandas as pd
import pickle as pickle
import itertools
#from datetime import datetime
import datetime

def AmazonMovies_BeforeDATE(num_reviews, file_name="movies.txt", outputTo='screen'):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    *6) review/time: 1182729600    (From the first 100K reviews >> max_date: 2012-10-25 min_date: 1997-12-19)
    7) review/summary: "There Is So Much Darkness Now ~ Come For The Miracle"
    8) review/text: Synopsis: On the daily..............
    9) null
    '''
    i = 0
    #single_entry=np.zeros(4, str)
    #single_entry_completed = True

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
            #print('line')
            i += 1
            #print(line)
            #for each entry, repeats at every 9
            if (i % 9 == 1):
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # !!!!
                # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                # that's why I added random numbers to the end because some userIDs are 14 letters...
                # !!!!
                single_entry=np.array([["productId....", "userId67891234", "profileName", "score"]])
                #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                single_entry[0, 0] = line[19:].rstrip("\n")
                #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
            elif (i % 9 == 2):
                single_entry[0, 1] = line[15:].rstrip("\n")
                #print('i={}  user={}'.format(i, line[15:].rstrip("\n")))
            elif (i % 9 == 6):
                # before, we used to take profilename (line[20:]) but now we are taking the time
                single_entry[0, 2] = line[13:].rstrip("\n")
            elif (i % 9 == 5):
                single_entry[0, 3] = line[14:].rstrip("\n")
            elif (i % 9 == 0):
                num_rec += 1
                #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                movie_data = np.append(movie_data, single_entry, axis = 0)
                #print('i={}  user={}'.format(i, movie_data))

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
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))
        
#def AmazonMovies_DATE(num_limit, num_reviews, date_year=2009, file_name="movies.txt", outputTo='screen'):
def AmazonMovies(num_limit, num_reviews, date_year=2009, file_name="movies.txt", outputTo='screen'):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    *6) review/time: 1182729600    (From the first 100K reviews >> max_date: 2012-10-25 min_date: 1997-12-19)
    7) review/summary: "There Is So Much Darkness Now ~ Come For The Miracle"
    8) review/text: Synopsis: On the daily..............
    9) null
    '''
    i = 0
    #single_entry=np.zeros(4, str)
    #single_entry_completed = True

    movie_data = np.array([["product/productId", "review/userId", "review/profileName", "review/score"]])
    #movie_data = np.array([["product/productId", "review/userId", "review/profileName", \
     #                       "review/helpfulness", "review/score"]])
    num_rec = 0
    num_rec_year = 0

    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown
    valid_record = True
    max_date = datetime.date(2000, 10, 5)
    min_date = datetime.date(2019, 10, 5)
    year_2009 = 0
    year_2010 = 0
    year_2011 = 0
    year_2012 = 0
    print('num_limit={}, num_reviews={}, date_year={}'.format(num_limit, num_reviews, date_year))
    
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            #print('line')
            i += 1
            #print(line)
            #for each entry, repeats at every 9
            if i < 1274406:
                valid_record = True
            else:
                if 'product/productId' in line:
                    valid_record = True
                    i += 9 - (i%9) + 1
                else:
                    valid_record = False
            
            # let's make this valid_record check better.. it looks a bit confusing..
            if (i % 9 == 1 and valid_record == True):
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # !!!!
                # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                # that's why I added random numbers to the end because some userIDs are 14 letters...
                # !!!!
                single_entry=np.array([["productId....", "userId67891234", "profileName", "score"]])
                #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                single_entry[0, 0] = line[19:].rstrip("\n")
                #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
            elif (i % 9 == 2):
                single_entry[0, 1] = line[15:].rstrip("\n")
                #print('i={}  user={}'.format(i, line[15:].rstrip("\n")))
            elif (i % 9 == 6):
                # before, we used to take profilename (line[20:]) but now we are taking the time
                # single_entry[0, 2] = line[13:].rstrip("\n")  # >> this was in use !!!!!
                try:
                    ts = int(line[13:].rstrip("\n"))
                    #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                    ts_datetime = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                    single_entry[0, 2] = ts_datetime
                except:
                    print('i={} .. ts={} .. num_rec={}'.format(i, ts, num_rec))
            elif (i % 9 == 5):
                single_entry[0, 3] = line[14:].rstrip("\n")
            elif (i % 9 == 0):
                if single_entry[0, 0] != "productId...." and single_entry[0, 1] != "userId67891234" and single_entry[0, 2] != "profileName" and single_entry[0, 3] != "score":
                    num_rec += 1
                    ts = datetime.date(*(int(s) for s in single_entry[0, 2].split('-')))   
                    if ts.year == date_year:
                       #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                        num_rec_year += 1
                        movie_data = np.append(movie_data, single_entry, axis = 0)
                        ts = datetime.date(*(int(s) for s in single_entry[0, 2].split('-')))
                        #print(ts.year)
                        #isGreater = True if ts > max_date else False
                        if ts > max_date:
                            max_date = ts
                        if ts < min_date:
                            min_date = ts

                        if ts.year == 2009:
                            year_2009 += 1
                        elif ts.year == 2010:
                            year_2010 += 1
                        elif ts.year == 2011:
                            year_2011 += 1
                        elif ts.year == 2012:
                            year_2012 += 1
                        #print('i={}  user={}'.format(i, movie_data))
                
            if (num_rec == num_limit or num_rec_year == num_reviews):
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
        #for movie in movie_data:
         #   print(str(movie) + "\n")
        print('max_date: {} min_date: {}'.format(max_date, min_date))
        print('2009: {} .. 2010: {} .. 2011: {} .. 2012: {}'.format(year_2009, year_2010, year_2011, year_2012))
    elif outputTo == 'dictionary':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        print('max_date: {} min_date: {}'.format(max_date, min_date))
        print('2009: {} .. 2010: {} .. 2011: {} .. 2012: {}'.format(year_2009, year_2010, year_2011, year_2012))
        
        return movie_dict
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))
    
def Filter_AmazonMovies(num_reviews, userID, movieID, file_name="movies.txt", outputTo='screen'):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    *3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    6) review/time: 1182729600   (From the first 100K reviews >> max_date: 2012-10-25 min_date: 1997-12-19)
    7) review/summary: "There Is So Much Darkness Now ~ Come For The Miracle"
    8) review/text: Synopsis: On the daily..............
    9) null
    '''
    i = 0
    #single_entry=np.zeros(4, str)
    single_entry_completed = True

    movie_data = np.array([["product/productId", "review/userId", "review/profileName", "review/score", "time", "text_sum"]])
    #movie_data = np.array([["product/productId", "review/userId", "review/profileName", \
     #                       "review/helpfulness", "review/score"]])
    num_rec = 0
    user_match = False
    movie_match = False
    
    print('movieID={}'.format(movieID))
    print('userID={}'.format(userID))
    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            #print('line')
            i += 1
            #print(line)
            #for each entry, repeats at every 9
            if (i % 9 == 1):
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # !!!!
                # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                # that's why I added random numbers to the end because some userIDs are 14 letters...
                # !!!!
                single_entry=np.array([["productId.....", "userId67891234", "profileName", "score", "time", "text_sum"]])
                #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                movie_name = line[19:].rstrip("\n")
                single_entry[0, 0] = movie_name
                #if movie_name == movieID:
                if movieID in movie_name:
                    movie_match = True
                    #print('i={} .. movie_name={}'.format(i, movie_name))
                #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
            elif (i % 9 == 2):
                user_name = line[15:].rstrip("\n")
                #print(user_name)
                single_entry[0, 1] = user_name
                #if user_name == userID:
                if userID in user_name:
                    user_match = True
                    #print('i={} .. user_name={}'.format(i, user_name))
                    #print(single_entry[0, 1])
                #print('i={}  user={}'.format(i, line[15:].rstrip("\n")))
            elif (i % 9 == 3):
                single_entry[0, 2] = line[20:].rstrip("\n")
            elif (i % 9 == 5):
                single_entry[0, 3] = line[14:].rstrip("\n")
            elif (i % 9 == 6):
                ts = int(line[13:].rstrip("\n"))
                ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                single_entry[0, 4] = ts_datetime
                #print(line[13:])
            elif (i % 9 == 8):
                # we want to check if the text is same or not. first 10 letters should be fine for that
                single_entry[0, 5] = line[13:].rstrip("\n")[0:10]
                #print(line[13:].rstrip("\n")[0:10])
            elif (i % 9 == 0):
                num_rec += 1
                #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                if user_match == True and movie_match == True:
                    movie_data = np.append(movie_data, single_entry, axis = 0)
                    
                user_match = False
                movie_match = False
                #print('i={}  user={}'.format(i, movie_data))

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
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))
        
def Filter_AmazonMovies_2(num_reviews, userID, movieID, file_name="movies.txt", outputTo='screen'):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    *3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    *6) review/time: 1182729600   (From the first 100K reviews >> max_date: 2012-10-25 min_date: 1997-12-19)
    7) review/summary: "There Is So Much Darkness Now ~ Come For The Miracle"
    8) review/text: Synopsis: On the daily..............
    9) null
    '''
    i = 0
    #single_entry=np.zeros(4, str)
    single_entry_completed = True

    movie_data = np.array([["product/productId", "review/userId", "review/profileName", "review/score", "time", "text_sum"]])
    #movie_data = np.array([["product/productId", "review/userId", "review/profileName", \
     #                       "review/helpfulness", "review/score"]])
    num_rec = 0
    user_match = False
    movie_match = False
    
    print('movieID={}'.format(movieID))
    print('userID={}'.format(userID))
    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown
    max_date = datetime.date(2000, 10, 5)
    min_date = datetime.date(2019, 10, 5)
    year_2009 = 0
    year_2010 = 0
    year_2011 = 0
    year_2012 = 0
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            #print('line')
            i += 1
            #print(line)
            #for each entry, repeats at every 9
            if (i % 9 == 1):
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # !!!!
                # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                # that's why I added random numbers to the end because some userIDs are 14 letters...
                # !!!!
                single_entry=np.array([["productId.....", "userId67891234", "profileName", "score", "time", "text_sum"]])
                #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                movie_name = line[19:].rstrip("\n")
                single_entry[0, 0] = movie_name
                #if movie_name == movieID:
                if movieID in movie_name:
                    movie_match = True
                    #print('i={} .. movie_name={}'.format(i, movie_name))
                #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
            elif (i % 9 == 2):
                user_name = line[15:].rstrip("\n")
                #print(user_name)
                single_entry[0, 1] = user_name
                #if user_name == userID:
                if userID in user_name:
                    user_match = True
                    #print('i={} .. user_name={}'.format(i, user_name))
                    #print(single_entry[0, 1])
                #print('i={}  user={}'.format(i, line[15:].rstrip("\n")))
            elif (i % 9 == 3):
                single_entry[0, 2] = line[20:].rstrip("\n")
            elif (i % 9 == 5):
                single_entry[0, 3] = line[14:].rstrip("\n")
            elif (i % 9 == 6):
                ts = int(line[13:].rstrip("\n"))
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                ts_datetime = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                single_entry[0, 4] = ts_datetime
                #print(line[13:])
            elif (i % 9 == 8):
                # we want to check if the text is same or not. first 10 letters should be fine for that
                single_entry[0, 5] = line[13:].rstrip("\n")[0:10]
                #print(line[13:].rstrip("\n")[0:10])
            elif (i % 9 == 0):
                num_rec += 1
                #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                if user_match == True and movie_match == True:
                    # 05.10.2009
                    #d_check = datetime.date(2009, 10, 5)
                    ts = datetime.date(*(int(s) for s in single_entry[0, 4].split('-')))
                    #print(ts.year)
                    #isGreater = True if ts > max_date else False
                    if ts > max_date:
                        max_date = ts
                    if ts < min_date:
                        min_date = ts
                        
                    if ts.year == 2009:
                        year_2009 += 1
                    elif ts.year == 2010:
                        year_2010 += 1
                    elif ts.year == 2011:
                        year_2011 += 1
                    elif ts.year == 2012:
                        year_2012 += 1
                    #print('Entry: {} .. date: {}'.format(single_entry, isGreater))
                    movie_data = np.append(movie_data, single_entry, axis = 0)
                    
                user_match = False
                movie_match = False
                #print('i={}  user={}'.format(i, movie_data))

            #if (num_rec == num_reviews):
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
        #print('screen output')
        print('max_date: {} min_date: {}'.format(max_date, min_date))
        print('2009: {} .. 2010: {} .. 2011: {} .. 2012: {}'.format(year_2009, year_2010, year_2011, year_2012))
        #for movie in movie_data:
         #   print(str(movie) + "\n")
    elif outputTo == 'dictionary':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        return movie_dict
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))
        
# we will make a better day filter later
def Filter_AmazonMovies_By_DATE(num_reviews, userID, movieID, date_filter='2009-', file_name="movies.txt", outputTo='screen', debug_mode='Off'):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    *3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    *6) review/time: 1182729600   (From the first 100K reviews >> max_date: 2012-10-25 min_date: 1997-12-19)
    7) review/summary: "There Is So Much Darkness Now ~ Come For The Miracle"
    8) review/text: Synopsis: On the daily..............
    9) null
    '''
    i = 0
    #single_entry=np.zeros(4, str)
    single_entry_completed = True

    movie_data = np.array([["product/productId", "review/userId", "review/profileName", "review/score", "time", "text_sum"]])
    #movie_data = np.array([["product/productId", "review/userId", "review/profileName", \
     #                       "review/helpfulness", "review/score"]])
    num_rec = 0
    user_match = False
    movie_match = False
    date_match = False
    
    print('movieID={}'.format(movieID))
    print('userID={}'.format(userID))
    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown

    valid_record = True
    max_date = datetime.date(2000, 10, 5)
    min_date = datetime.date(2019, 10, 5)
    year_2009 = 0
    year_2010 = 0
    year_2011 = 0
    year_2012 = 0
    
    print('num_limit={}, num_reviews={}, date_filter={}'.format(num_limit, num_reviews, date_filter))
    
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            #print('line')
            i += 1
            #print(line)
            #for each entry, repeats at every 9
            
            if i < 1274406:
                valid_record = True
            else:
                if 'product/productId' in line:
                    valid_record = True
                    i += 9 - (i%9) + 1
                else:
                    valid_record = False
            
            # let's make this valid_record check better.. it looks a bit confusing..
            if (i % 9 == 1 and valid_record == True):
                # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                #single_entry = np.zeros(4, str)
                # !!!!
                # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                # that's why I added random numbers to the end because some userIDs are 14 letters...
                # !!!!
                single_entry=np.array([["productId....", "userId67891234", "profileName", "score"]])
                #single_entry=np.array([["productId.....", "userId67891234", "profileName", "score", "time", "text_sum"]])
                #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                movie_name = line[19:].rstrip("\n")
                single_entry[0, 0] = movie_name
                #if movie_name == movieID:
                if movieID in movie_name:
                    movie_match = True
                    #print('i={} .. movie_name={}'.format(i, movie_name))
                #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
            elif (i % 9 == 2):
                user_name = line[15:].rstrip("\n")
                #print(user_name)
                single_entry[0, 1] = user_name
                #if user_name == userID:
                if userID in user_name:
                    user_match = True
                    #print('i={} .. user_name={}'.format(i, user_name))
                    #print(single_entry[0, 1])
                #print('i={}  user={}'.format(i, line[15:].rstrip("\n")))
            elif (i % 9 == 6):
                ts = int(line[13:].rstrip("\n"))
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                #ts_datetime = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                if date_filter in ts:
                    date_match = True
                ts_datetime = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                single_entry[0, 4] = ts_datetime
                #print(line[13:])
            elif (i % 9 == 5):
                single_entry[0, 3] = line[14:].rstrip("\n")
            elif (i % 9 == 0):
                if single_entry[0, 0] != "productId...." and single_entry[0, 1] != "userId67891234" and single_entry[0, 2] != "profileName" and single_entry[0, 3] != "score" and user_match == True and movie_match == True and date_match == True:
                    num_rec += 1
                    ts = datetime.date(*(int(s) for s in single_entry[0, 2].split('-')))   
                    if ts.year == date_year:
                       #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                        num_rec_year += 1
                        movie_data = np.append(movie_data, single_entry, axis = 0)
                        ts = datetime.date(*(int(s) for s in single_entry[0, 2].split('-')))
                        #print(ts.year)
                        #isGreater = True if ts > max_date else False
                        
                        if debug_mode == 'On':
                            if ts > max_date:
                                max_date = ts
                            if ts < min_date:
                                min_date = ts
                            if ts.year == 2009:
                                year_2009 += 1
                            elif ts.year == 2010:
                                year_2010 += 1
                            elif ts.year == 2011:
                                year_2011 += 1
                            elif ts.year == 2012:
                                year_2012 += 1
                        
                        #print('i={}  user={}'.format(i, movie_data))            
                    
                user_match = False
                movie_match = False
                date_match = False
                #print('i={}  user={}'.format(i, movie_data))

            #if (num_rec == num_reviews):
            if (num_rec == num_limit or num_rec_year == num_reviews):
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
        #print('screen output')
        print('max_date: {} min_date: {}'.format(max_date, min_date))
        if debug_mode == 'On':
            print('2009: {} .. 2010: {} .. 2011: {} .. 2012: {}'.format(year_2009, year_2010, year_2011, year_2012))
        #for movie in movie_data:
         #   print(str(movie) + "\n")
    elif outputTo == 'dictionary':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        return movie_dict
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))

def AmazonMovies_VALIDATION(num_reviews, file_name="movies.txt", outputTo='screen', start_index = 40000):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    *6) review/time: 1182729600
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
    
    remain_start_index = start_index % 9
    if remain_start_index != 1:
        start_index += remain_start_index + 1
    
    #print(start_index)
        
    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:
            #print(line)
            i += 1
            # we don't want to read the first records because we use them to build our model
            # we will use this batch for validation
            if i > start_index * 9:
                #for each entry, repeats at every 9
                if (i % 9 == 1):
                    # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                    #single_entry = np.zeros(4, str)
                    # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                    #single_entry = np.zeros(4, str)
                    # !!!!
                    # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                    # that's why I added random numbers to the end because some userIDs are 14 letters...
                    # !!!!
                    single_entry=np.array([["productId....", "userId67891234", "profileName", "score"]])
                    #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                    #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
                    single_entry[0, 0] = line[19:].rstrip("\n")
                elif (i % 9 == 2):
                    single_entry[0, 1] = line[15:].rstrip("\n")
                    #print('i={}  user={}'.format(i, line[15:].rstrip("\n")))
                elif (i % 9 == 6):
                    #elif (i % 9 == 3):
                    # before, we used to take profilename (line[20:]) but now we are taking the time
                    single_entry[0, 2] = line[13:].rstrip("\n")
                elif (i % 9 == 5):
                    #print('i={}.. line={}'.format(i, line))
                    #print(single_entry)
                    #print(type(single_entry))
                    #print(single_entry.shape)
                    single_entry[0, 3] = line[14:].rstrip("\n")
                elif (i % 9 == 0):
                    num_rec += 1
                    #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                    #print('i={}.. line={}'.format(i, line))
                    movie_data = np.append(movie_data, single_entry, axis = 0)
                    #print('i={}  user={}'.format(i, movie_data))

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
        
        print('Your file saved as {}'.format(file_name))
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
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))
        
def AmazonMovies_VALIDATION_DEBUG(num_reviews, file_name="movies.txt", outputTo='screen', start_index = 40000):
    '''
    *1) product/productId: B003AI2VGA 
    !!!!! >>> some MOVIES don't start with B but instead full numbers like: 0790747324   <<< !!!!!
    *2) review/userId: A141HP4LYPWMSR
    3) review/profileName: Brian E. Erland "Rainbow Sphinx"
    4) review/helpfulness: 7/7
    *5) review/score: 3.0
    *6) review/time: 1182729600
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
    
    remain_start_index = start_index % 9
    
    print('start_index: {}'.format(start_index))
    print('remain_start_index: {}'.format(remain_start_index))
    
    if remain_start_index != 1:
        start_index += remain_start_index + 2
    
    ## ***** We don't necessarilly need helpfulness, just keeping it in case we use it, otherwise, we can remove it
    # encoding="latin-1 >> is needed otherwise possible to receive:
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0x9c
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    # reading/writing the file with "with"is also important since "with"handles opening/closing the file
    # after finishing its job or even if there is an exception thrown
    line_no=0
    first_record = False
    with open(file_name, encoding="latin-1") as input_file:
        for line in input_file:            
            i += 1
            # we don't want to read the first records because we use them to build our model
            # we will use this batch for validation
            #print('line=\n{}'.format(line))
            
            if 'product/productId' in line:
                first_record = True
                i += 9 - (i%9) + 1          
            
            if i > start_index * 9 and first_record == True:
                #print('i='+str(i))
                #print('i_remainder={}'.format(i%9))
                
                print('line_no={}'.format(line_no))
                print('***********')
                print('line={}'.format(line))
                line_no+=1
                #i += 8
                #print(i)
                #print('line=\n{}'.format(line))
                #for each entry, repeats at every 9
                if (i % 9 == 1):
                    # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                    #single_entry = np.zeros(4, str)
                    # just to be sure we are not carrying anything from the previous record, we reset the single_entry
                    #single_entry = np.zeros(4, str)
                    # !!!!
                    # if we just create below singleentry with "userId", it trims the userID to 11 letters...
                    # that's why I added random numbers to the end because some userIDs are 14 letters...
                    # !!!!
                    single_entry=np.array([["productId....", "userId67891234", "profileName", "score"]])
                    #single_entry=np.array([["productId", "userId", "profileName", "helpfulness", "score"]])
                    #print('i={}  product={}'.format(i,line[19:].rstrip("\n")))
                    single_entry[0, 0] = line[19:].rstrip("\n")
                    #single_entry[0] = line[19:].rstrip("\n")
                    #print('i={} .. i_r={} ..  product={}'.format(i, i%9, single_entry[0, 0]))
                    print('i={} .. i_r={} ..  product={}'.format(i, i%9, single_entry[0, 0]))
                    #first_record = True
                elif (i % 9 == 2):
                    #single_entry[0, 1] = line[15:].rstrip("\n")
                    #single_entry=np.array([["productId....", "userId67891234", "profileName", "score"]])
                    single_entry[0, 1] = line[15:].rstrip("\n")
                    print('i={} .. i_r={} .. user={}'.format(i, i%9, single_entry[0, 1]))
                    #print('i={} .. i_r={} .. user={}'.format(i, i%9, single_entry[1]))
                elif (i % 9 == 6):
                    #elif (i % 9 == 3):
                    # before, we used to take profilename (line[20:]) but now we are taking the time
                    single_entry[0, 2] = line[13:].rstrip("\n")
                    #single_entry[2] = line[13:].rstrip("\n")
                    print('i={} .. i_r={} .. time={}'.format(i, i%9, single_entry[0, 2]))
                    #print('i={} .. i_r={} .. time={}'.format(i, i%9, single_entry[2]))
                elif (i % 9 == 5):
                    #print('i={}.. line={}'.format(i, line))
                    #print(single_entry)
                    #print(type(single_entry))
                    #print(single_entry.shape)
                    single_entry[0, 3] = line[14:].rstrip("\n")
                    #single_entry[3] = line[14:].rstrip("\n")
                    print('i={} .. i_r={} .. score={}'.format(i, i%9, single_entry[0, 3]))
                    #print('i={} .. i_r={} .. score={}'.format(i, i%9, single_entry[3]))
                elif (i % 9 == 0):
                    num_rec += 1
                    #['B003AI2VGA\n' 'A141HP4LYPW' 'Brian E. Er' '3.0\n']
                    #print('i={}.. line={}'.format(i, line))
                    print('single_entry:{}'.format(single_entry))
                    print('i={} .. i_r={} .. line'.format(i, i%9))
                    print(line)
                    movie_data = np.append(movie_data, single_entry, axis = 0)
                    #print('i={}  user={}'.format(i, movie_data))

                #if (num_rec == num_reviews):
                if (line_no == num_reviews):
                    fileinput.close()
                    break

    if outputTo == 'file':
        # to avoid overwriting an existing file, if you want to overwrite, just basically use output.txt
        file_name = FileNameUnique() 
        #with open("output.txt", "w") as output:
        with open(file_name, "w") as output:
            for movie in movie_data:
                output.write(str(movie) + "\n")
        
        print('Your file saved as {}'.format(file_name))
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
    elif outputTo == 'dict_to_file':
        movie_dict = {}
        key = 0
        for movie in movie_data:
            movie_dict[key] = [movie[0], movie[1], movie[2], movie[3]]
            key += 1
        del movie_dict[0]
        
        # saves as a binary file
        file_name = FileNameUnique(suffix = '.pkl')
        print('saving file')
        with open(file_name, 'wb') as f:
            pickle.dump(movie_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Your file saved as {}'.format(file_name))

def Time_Stamp():
    date_time_now = datetime.datetime.now()
    
    D = str(date_time_now.day)
    M = str(date_time_now.month)
    Y = str(date_time_now.year)

    h = str(date_time_now.hour)
    m = str(date_time_now.minute)
    s = str(date_time_now.second)
    
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

def Read_Data_From_Raw_Save(file_name):
    #lst_tst = amp.Read_Connected_Movies('Movies_22.5.2020_20.47.56.txt')
    lst_tst = amp.Read_Connected_Movies(file_name)
    arr_tst = lst_tst[0][0].replace('[','').replace(']','').replace("'",'').split(sep=" ")
    
def Load_Pickle_File_x(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
            
def Load_Pickle_File(file_name, n_reviews = 20):
    with open(file_name, 'rb') as f:
        dict_movies = pickle.load(f)
    
    #import itertools
    return dict(itertools.islice(dict_movies.items(), n_reviews))

def Load_Pickle_File_VAL(file_name, start_index, n_reviews = 20):
    with open(file_name, 'rb') as f:
        dict_movies = pickle.load(f)
    
    #import itertools
    return dict(list(dict_movies.items())[start_index:start_index+n_reviews])
    