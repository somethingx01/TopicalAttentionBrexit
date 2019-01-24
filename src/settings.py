# -*- coding: utf8 -*-

#ROOT_DIR = '.' #the root path, currently is the code execution path
ROOT_DIR = '/media/data1/TopicalAttentionBrexit'

TRAIN_SET_PATH = '%s/datasets/BrexitTweets.snstrain_08.wordembbed.csv'%ROOT_DIR
TEST_SET_PATH = '%s/datasets/BrexitTweets.snstrain_00.wordembbed.csv'%ROOT_DIR

SAVE_DIR = '%s/save'%ROOT_DIR

WORD_EMBEDDING_DIMENSION = 300 #must set before run
TOPIC_EMBEDDING_DIMENSION = 10
#ALPHABET_SIZE=len(ALPHABET)
#DICT = {ch: ix for ix, ch in enumerate(ALPHABET)}
#CLASS_LEVEL1 = 17 #must set before run
#CLASS_LEVEL2 = 12 #must set before run
TWITTER_LENGTH = 24 #universal twitter length for each twitter, must set before run
USER_SELF_TWEETS = 3 #a user's previous tweet nums, must set before run
NEIGHBOR_TWEETS = 5 #neighbors' previous tweet nums, must set before run
TRAINING_INSTANCES = 10880
TESTING_INSTANCES = 10965

# CLASS_COUNT = 3 # number of classes for classification  
TOPIC_SENTIMENT_COUNT = 9
SENTIMENT_COUNT = 3

CHUNK_SIZE_MULTIPLIER_BATCH_SIZE = 2 # CHUNK_SIZE = CHUNK_SIZE_MULTIPLIER_BATCH_SIZE * BATCH_SIZE
CONST_TWEET_WEIGHT_A = 1000.0 # 1000 original, smaller better, more topical influence


class DefaultConfig():
    '''
    default config for training parameters
    '''
    batch_size = 1024 # 256 best for 01 #if if 2048 then cuda memory exceeds
    epochs = 100 #200 original
    learning_rate = 0.005 # 0.0005 initial best, learning rate initialize should depend on the batch size
    lr_decay = 0.93 # 0.93 0.005 best 
    weight_decay = 1e-4
    model = 'TopicalAttentionGRU'
    
    #on_cuda = False # if this is false then run CPU
    on_cuda = True # if this is false then run CPU

    # tdnn_kernel=[(1,25),
    #             (2,50),
    #             (3,75),
    #             (4,100),
    #             (5,125),
    #             (6,150),
    #             (7,175)],
    # highway_size=700,
    # rnn_hidden_size=650,
    # dropout':0.0
    def set_attrs(self,kwargs):
        '''
        kwargs is a dict
        '''
        for k,v in kwargs.items():
            setattr(self,k,v)#inbuilt function of python, set the attributes of a class object. For example: setattr(oDefaultConfig,epochs,50) <=> oDefaultConfig.epochs = 50
    
    def get_attrs(self):
        '''
        the enhanced getattr, returns a dict whose key is public items in an object
        '''
        attrs = {} #attrs = dict()
        for k , v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'set_attrs' and k != 'get_attrs' :
                attrs[k] = getattr( self , k) #get the attr in an object
        return attrs

if __name__=='__main__':
    config=DefaultConfig()
    print(config.get_attrs())
    config.set_attrs( { 'epochs':200 , 'batch_size' : 64 } )
    print(config.get_attrs())
