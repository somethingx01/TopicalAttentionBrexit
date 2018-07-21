# -*- coding: utf8 -*- 

import logging
#from settings import CLASS_LEVEL1
from settings import * #import all the variables and classes in .py
import settings #you shall write setting.CLASS_LEVEL1 to use the variable in .py
import os,sys
import pandas
import numpy

from math import ceil

def get_logger(name):
    '''
    set and return the logger modula,
    output: std
    '''
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def latest_save_num():
    if not os.path.exists( SAVE_DIR ):
        os.makedirs( SAVE_DIR )
    files = os.listdir( SAVE_DIR ) #list all the files in save dir
    maxno = -1
    # print(files)
    for f in files:
        if os.path.isdir(SAVE_DIR+'/'+f):
            # print(f,'true')
            try:
                # print(name)
                no=int(f)
                maxno=max(maxno,no)
                # print(maxno)
            except Exception as e:
                print( e.message ,file = sys.stderr)
                pass
    return maxno

class DataManager():
    '''
    data loading module
    '''
    #__df = None # not use this format of data initialization, possible confusion
    #__cursor = 0

    def __init__(self , 
        param_batch_size , 
        param_training_instances_size):
        #param_topic_sentiment_count,
        #param_sentiment_count ):
        #self.df=None
        #self.cursor=0
        #self.batch_size=batch_size 

        #global private variable initialized here
        
        self.__current_dataframe_of_pandas = None #dataframe read by a chunk
        self.__current_cursor_in_dataframe = 0 #the cursor shifted in pandas, it's the global cursor shift in dataframe
        self.__batch_size = param_batch_size
        #self.__topic_sentiment_count = param_topic_sentiment_count
        #self.__sentiment_count = param_sentiment_count
        self.__training_instances_size = param_training_instances_size

        #self.__batch_size = param_batch_size
        #self.__current_cursor = 0 #the current cursor in file, is the line num
        #self.__current_file_pointer = None list_all_tweets_of_auser#the current file pointer

    def load_dataframe_from_file( self, param_filepath_in):
        '''
        read once, initialize dataframe_of_pandas
        '''

        print('Loading dataframe...')
        self.__current_dataframe_of_pandas = pandas.read_csv( param_filepath_in, dtype = numpy.float32, header = None, encoding = 'utf-8',  sep = '\s+' , engine = 'c')
        
        #self.__dataframe_of_pandas = pandas.read_csv( param_filepath_in, header = None, encoding = 'utf8',  sep = '\t' , engine = 'c') # you can use regular expression in sep by setting engine = 'python'
        #engine = 'c' will face error, may be 'c' needs 0 0 0 to be 0.0 0.0 0.0

        #c do not support \t\n
        #print(len(self.__dataframe_of_pandas) )
        #currently every 9 lines describe a user
    
    def reshuffle_dataframe(self):
        self.__current_dataframe_of_pandas.sample( frac=1 )

    def next_batch(self):
        '''
        obtain next batch
        possible out of range, you have to control the tail
        the best way is to call next_batch dataframe_size()//batch_size times
        then call tail_batch
        use reshape
        '''

        batch_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        #dataframe_of_pandas = pandas.DataFrame( data= self.__chunks_of_pandas.get_chunk( chunk_size), dtype= numpy.float32) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        batch_index = s // batch_size

        print("Loading batch: %d" % (batch_index))
        #dataframe_of_pandas = self.__current_dataframe_of_pandas#.get_chunk( chunk_size) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        #print( 'get_chunk: '+ str( s//batch_size) )
        batch_x = numpy.zeros( ( batch_size, USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size, TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT ) )
        for user_i in range(s,t):
            #print( 'user_i: '+ str(user_i) )
            #each user's time serial
            label_shift = TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT #one col for label
            
            # print( label_shift)
            
            #batch_x[ user_i%batch_size] = numpy.reshape( dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift: ], (1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION) )
            #batch_x[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift: ].values.reshape( ( 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION) )

            # dataframe_of_pandas infact is a dataframe of a chunk_size
            batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ label_shift: ].values.reshape( ( USER_SELF_TWEETS, NEIGHBOR_TWEETS+1, TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
            batch_y[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ 0: label_shift] #iloc is absolute shift, loc is access by row name. if header==None, then the row name is the int index, which is inherented by splitted chunks
            #print(batch_x.shape)

            # list_labels = self.__dataframe_of_pandas.iloc[ user_i , 0 ]
            # batch_y[ user_i%batch_size ][ 0 ] = list_labels[ 0 ]
            #batch_y[ user_i%batch_size ][ 1 ] = list_labels[ 1 ]

        self.__current_cursor_in_dataframe = t

        return batch_x,batch_y

    def set_current_cursor_in_dataframe_zero(self):
        '''
        if INSTANCE % batch_size == 0, then the tail_batch won't be called, so call this function to reset the __cursor_in_current_frame
        '''
        self.__current_cursor_in_dataframe = 0

    def tail_batch( self):

        batch_size= self.__batch_size

        s= self.__current_cursor_in_dataframe
        t= s + batch_size

        batch_index = s // batch_size

        print("Loading batch: %d"%(batch_index))


        #self.__current_dataframe_of_pandas#.get_chunk( chunk_size) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        batch_x = numpy.zeros( ( batch_size, USER_SELF_TWEETS, 1 + NEIGHBOR_TWEETS , TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size, TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT) )

        #complement the last chunk with the initial last chunk
        # assert len(self.__current_dataframe_of_pandas) == self.__training_instances_size
        last_batch_size = len( self.__current_dataframe_of_pandas) % batch_size
        
        append_times = batch_size // last_batch_size
        append_tail = batch_size % last_batch_size

        #dataframe_of_pandas_compensated = pandas.DataFrame(data = None, columns = dataframe_of_pandas.axes[1] )
        
        #for i in range( 0, append_times) :
        #    dataframe_of_pandas_compensated= dataframe_of_pandas_compensated.append( dataframe_of_pandas.iloc[ s : s + last_batch_size ], ignore_index=True) # for ignore_index refer to the manual

        #dataframe_of_pandas_compensated= dataframe_of_pandas_compensated.append( dataframe_of_pandas.iloc[ s: s + append_tail], ignore_index=True)

        for user_i in range(s,t):
            #list_all_tweets_of_auser = self.__dataframe_of_pandas.iloc[ i , 2:(2+1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT) ].values.tolist()
            #list_a            label_shift = 1 #one col for label
            label_shift = TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT #one col for label

            #batch_x[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift: ].reshape( (1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION) )
            if (user_i % batch_size) < last_batch_size:    
                batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ label_shift: ].values.reshape( ( USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH*WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
                batch_y[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][0 : label_shift]
            else:
                shift_in_last_batch_size = (user_i % batch_size) % last_batch_size
                batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i - user_i % batch_size + shift_in_last_batch_size][ label_shift: ].values.reshape( ( USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH*WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
                batch_y[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i - user_i % batch_size + shift_in_last_batch_size][0 : label_shift]


        
        self.__current_cursor_in_dataframe = 0

        return batch_x,batch_y

    def n_batches(self ):
        # if self.__current_dataframe_of_pandas == None:
        #     print( 'Error: __current_dataframe_of_pandas == None' , file = sys.stderr)
        # else:
        return ceil( self.__training_instances_size / self.__batch_size )

if __name__ == '__main__':
    # print( os.getcwd() )
    oDataManager = DataManager( param_batch_size = 128 , param_training_instances_size = TRAINING_INSTANCES )
    
    # oDataManager.generate_csv_from_wordembbed( TRAIN_SET_PATH )
    
    oDataManager.load_dataframe_from_file( TRAIN_SET_PATH)

    # print(oDataManager.dataframe_size() // 64 )
    # print(oDataManager.dataeframe_size() % 64 )
    
    print('Hello')
    for i in range(0, TRAINING_INSTANCES// 128) :
        print(i)
        (batch_x,batch_y) = oDataManager.next_batch()
        print('shape_x:' , batch_x.shape , '--shape_y:' , batch_y.shape ) 
    ( batch_x, batch_y ) = oDataManager.tail_batch()
    print( batch_x.shape ) #size of torch in numpy
    print( batch_y.shape )