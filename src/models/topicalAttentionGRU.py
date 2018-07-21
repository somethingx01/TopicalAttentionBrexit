# -*- coding: utf8 -*-

import torch

import torch.nn #torch.nn and torch two different module from torch
import torch.nn.functional

import torch.autograd

from models.attention import Attention
#from attention import Attention

class TopicalAttentionGRU(torch.nn.Module):
    '''
    You have to inherent nn.Module to define your own model
    two virtual functions, loda and save must be instantialized
    '''
    def __init__(self,
        param_word_embed_size,# 300 in our model
        param_topic_embed_size, #it depends on the setting
        param_fix_sentence_length, # depends on the setting
        param_rnn_tweet_hidden_size, # the size of tweet level rnn
        param_attention_size, # the attention size
        param_user_self_tweets, # setting
        param_neighbor_tweets, #in the setting 
        param_constant_tweet_weight_a,
        param_rnn_hidden_size,
        param_class_count,
        param_on_cuda): # class of user labels, in fact it is the class of level 1 labels

        super(TopicalAttentionGRU,self).__init__()
        self.modelname='TopicalAttentionGRU' #same with the class name
    
        self.word_embed_size = param_word_embed_size
        self.topic_embed_size = param_topic_embed_size
        self.fix_sentence_length = param_fix_sentence_length
        self.attention_size = param_attention_size
        self.user_self_tweets = param_user_self_tweets
        self.neighbor_tweets = param_neighbor_tweets
        self.constant_tweet_weight_a = param_constant_tweet_weight_a
        self.class_count = param_class_count
        if param_on_cuda:
            self.ids_only_wordembed_dim = torch.autograd.Variable( torch.LongTensor( [ i for i in range( 0 , self.fix_sentence_length * self.word_embed_size ) ] ) ).cuda()
            self.ids_only_topic_embedding = torch.autograd.Variable( torch.LongTensor( [ i for i in range( self.fix_sentence_length * self.word_embed_size, self.fix_sentence_length * self.word_embed_size + self.topic_embed_size ) ] ) ).cuda()
        else:
            self.ids_only_wordembed_dim = torch.autograd.Variable( torch.LongTensor( [ i for i in range( 0 , self.fix_sentence_length * self.word_embed_size ) ] ) ).cpu()
            self.ids_only_topic_embedding = torch.autograd.Variable( torch.LongTensor( [ i for i in range( self.fix_sentence_length * self.word_embed_size, self.fix_sentence_length * self.word_embed_size + self.topic_embed_size ) ] ) ).cpu()

        self.rnn_tweet_hidden_size = param_rnn_tweet_hidden_size
        self.rnn_tweet_layer_count = 1
        self.rnn_tweet = torch.nn.LSTM( self.word_embed_size, self.rnn_tweet_hidden_size, self.rnn_tweet_layer_count, dropout = 0.0, batch_first=True, bidirectional=False )
        
        if param_on_cuda:
            self.ids_seq_last = torch.autograd.Variable( torch.LongTensor( [ self.user_self_tweets -1 ] ) ).cuda()
        else:
            self.ids_seq_last = torch.autograd.Variable( torch.LongTensor( [ self.user_self_tweets -1 ] ) ).cpu()

        self.linear_tweet = torch.nn.Linear( self.rnn_tweet_hidden_size, self.word_embed_size)
        # mean pooling is functioned as mean after the LSTM

        self.A_alpha = torch.nn.Parameter( torch.Tensor( ( self.fix_sentence_length ) ) )
        # self.B_alpha = torch.nn.Parameter( torch.Tensor( ( self.word_embed_size ) ) ) # you can choose this to be a vector, but this may cause Curse of Dimensionality
        self.B_alpha = torch.nn.Parameter( torch.Tensor( ( 1 ) ) ) # can use a single value or matrix

        self.A_alpha.data.normal_(std=0.1)
        self.B_alpha.data.normal_(std=0.1)

        #==========The original for Attention Layer
        self.A_tweets = torch.nn.Parameter( torch.Tensor( ( self.neighbor_tweets + 1 ) ) ) # attention for twitter embed
        self.B_tweets = torch.nn.Parameter( torch.Tensor( ( 1 ) ) ) # attention for twitter embed

        self.A_tweets.data.normal_(std=0.1)
        self.B_tweets.data.normal_(std=0.1)
        #==========The original for Attention Layer

        #==========The latest for Attention Layer
        # self.attention_over_tweets = Attention(self.attention_size, self.word_embed_size + self.topic_embed_size ) # to handle biLSTM output

        #==========The latest for Attention Layer

        
        
        #the attention size can be batch-dependent, that is the case in emoj_LSTM, but there we think for all the input the attention is the same

        self.rnn_hidden_size = param_rnn_hidden_size # hidden size: the vector size of output for each time, It's a vector!
        self.rnn_layer_count = 1 #constant, I haven't implemented the dynamic programming yet, but it defines the RNN LayerNum
        self.rnn = torch.nn.GRU( self.word_embed_size + self.topic_embed_size , self.rnn_hidden_size ,self.rnn_layer_count , dropout = 0.0 , batch_first = True , bidirectional = False ) # batchfirst: the datasize is [ batch_size, seq , feature ]
        self.linear = torch.nn.Linear( self.rnn_hidden_size , self.class_count )
        self.logsoftmax = torch.nn.LogSoftmax( dim = 1) # dim= 0 means sum( a[i][1][3]) = 1


    def load(self , path):
        self.load_state_dict( torch.load(path) )

    def save(self, path):
        save_result = torch.save( self.state_dict() , path )
        return save_result
        
    def save_full(self, path):
        save_result = torch.save( self, path )
        return save_result

    def forward( self , param_input ):
        '''
        from input to output
        '''
        ( batch_size , user_tweet_count , neighbor_tweet_count_add_one , twitter_length_size_x_word_embed_add_topic_embed_size ) = param_input.size()
        assert self.fix_sentence_length * self.word_embed_size + self.topic_embed_size == twitter_length_size_x_word_embed_add_topic_embed_size
        assert 1 + self.neighbor_tweets == neighbor_tweet_count_add_one
        assert self.user_self_tweets == user_tweet_count          

        var_only_wordembed_dim = param_input.index_select( 3 , self.ids_only_wordembed_dim ) #var only has word embed line
        var_only_topic_embedding = param_input.index_select( 3, self.ids_only_topic_embedding )
        
        #ids_only_first_word_of_topic_embedding = torch.autograd.Variable( torch.LongTensor( [ 0 ] ) )
        #var_only_first_word_of_topic_embedding = var_only_topic_embedding.index_select( 3 , ids_only_first_word_of_topic_embedding )

        #var_only_first_word_of_topic_embedding = var_only_first_word_of_topic_embedding.squeeze()
        
        #print( var_only_first_word_of_topic_embedding.size() )

        
        #var_only_wordembed_dim = var_only_wordembed_dim.view( batch_size, user_tweet_count, neighbor_tweet_count_add_one, self.fix_sentence_length, self.word_embed_size )
        #print(var_only_wordembed_dim.size())
        var_only_wordembed_dim = var_only_wordembed_dim.view( batch_size, user_tweet_count, neighbor_tweet_count_add_one, self.fix_sentence_length, self.word_embed_size )

        var_only_wordembed_dim = var_only_wordembed_dim.view( -1, self.fix_sentence_length, self.word_embed_size )

        var_only_wordembed_dim_permuted = var_only_wordembed_dim.permute( 1, 0, 2 ) #transpose
        
        #print( var_only_wordembed_dim_permuted.size() )

        #print( self.A_alpha.size() )

        #var_twitter_embedded = torch.mv( var_only_wordembed_dim_permuted , self.A_alpha ) #twitter embed # can only accept matrix and vector, no more than 2D
        #print( var_only_wordembed_dim_permuted.size( ) )
        var_rnn_tweet_output, (var_rnn_tweet_output_h, var_rnn_tweet_output_c) = self.rnn_tweet( var_only_wordembed_dim_permuted)

        #var_twitter_embedded = var_twitter_embedded.view( batch_size , timeserial_size , userandneighbor_size , self.word_embed_size , 1 ).permute( 0, 1, 2, 4, 3 )
        var_rnn_tweet_output = var_rnn_tweet_output.permute(1, 0, 2)
        
        #print( var_rnn_tweet_output.size( ) )

        var_twitter_embedded = torch.mean( var_rnn_tweet_output, dim=1 ) #default squeezed
        var_twitter_embedded = self.linear_tweet( var_twitter_embedded )
        var_twitter_embedded = var_twitter_embedded.view(batch_size, self.user_self_tweets, neighbor_tweet_count_add_one, self.word_embed_size )

        #print( var_twitter_embedded.size( ) )

        #get twitter attention

        
        var_twitter_embedded = var_twitter_embedded * self.constant_tweet_weight_a
        var_twitter_and_topic_embedded = torch.cat( ( var_twitter_embedded , var_only_topic_embedding ) ,dim = 3 ) 
        #var_twitter_embedded = var_twitter_embedded.mul( self.A_beta ) * self.constant_tweet_weight_a
        #var_only_first_word_of_topic_embedding = var_only_first_word_of_topic_embedding.mul( self.A_gamma )
        var_twitter_and_topic_embedded = var_twitter_and_topic_embedded.mul( 1.0/ ( self.constant_tweet_weight_a + 1) ) 
        
        #==========The original for tweet attention
        #var_twitter_emids_only_usercurr_dimbedded = var_twitter_embedded.squeeze()

        #print( var_twitter_and_topic_embedded.size( ) )
        var_twitter_and_topic_embedded = var_twitter_and_topic_embedded.permute( 0, 1, 3, 2 )
        var_user_tweet_context = torch.mv( var_twitter_and_topic_embedded.contiguous().view(-1 , self.neighbor_tweets + 1) , self.A_tweets) + self.B_tweets.expand( batch_size, user_tweet_count, self.word_embed_size + self.topic_embed_size ).contiguous().view(-1)

        #print( var_only_userprev_dim.size() )
        #print( var_only_neighborprev_dim.size() )

        #print( var_all_usercategory_embed.size() )
        #==========The original for tweet attention

        #==========The latest for tweet attention
        # #print(var_twitter_and_topic_embedded.size())
        # var_twitter_and_topic_embedded = var_twitter_and_topic_embedded.view(batch_size * user_tweet_count, neighbor_tweet_count_add_one,  self.word_embed_size + self.topic_embed_size  )
        # var_user_tweet_context = self.attention_over_tweets(var_twitter_and_topic_embedded)
        # var_user_tweet_context = var_user_tweet_context.view(batch_size, user_tweet_count, self.word_embed_size + self.topic_embed_size )
        # #print(var_user_tweet_context.size())
        #==========The latest for tweet attention

        var_user = var_user_tweet_context.view( batch_size, user_tweet_count, self.word_embed_size + self.topic_embed_size)

        #==========The original for permute
        var_user = var_user.permute( 1 , 0 , 2 )  # permute to (seq_len, batch, input_size)
        
        #print( var_timeserials_embed.size( ) )
        var_rnn_output , var_rnn_output_h  = self.rnn( var_user , None ) # None means that h_0 = 0
        #print( var_rnn_output.size() )
        
        var_rnn_output = var_rnn_output.permute( 1 , 0 , 2 )
        #==========The original for permute

        #==========The latest for permute
        # var_rnn_output , var_rnn_output_h  = self.rnn( var_user , None ) # None means that h_0 = 0
        #==========The latest for permute

        # <==> torch.index_select

        var_seq_last = var_rnn_output.index_select( 1 , self.ids_seq_last )

        #print( var_seq_last.size() )
        
        var_seq_last = var_seq_last.squeeze()
        
        #print( var_seq_last.size() )

        var_linear_output = self.linear( var_seq_last )

        #print( var_linear_output.size( ) )
        #print( var_linear_output.data[0] )
        #var_softmax_output = self.softmax( var_linear_output)
        var_logsoftmax_output = self.logsoftmax( var_linear_output )
        #print( var_softmax_output.data[0] )
        #print( var_softmax_output.size( ) )
        return var_logsoftmax_output


if __name__ == '__main__':
    # m = torch.randn(4,5,6)
    # print(m)userandneighbor_size
    # m_var = torch.autograd.Variable( m )
    # #ids = torch.Tensor([1,1,0,0]).long() #autograd = false by acquiescence
    # #var2 =  m.gather(1, ids.view(-1,1))
    # ids = torch.LongTensor( [ 2 , 4 ] )

    # ids_var = torch.autograd.Variable( ids )
    
    #they have the same function
    # var2 = m.index_select( 2 , ids  )
    # print( var2 )
    # var3 = torch.index_select( m , 2 , ids )
    # print( var3 )var_only_userprev_dim
    # var2_var = m_var.index_select( 2 , ids_var )
    # print(var2_var)

    # #model=model.cpu() #load the model to the CPU, 2 different ways to load the model
    # #model=model.cuda() #load the model to the GPU

    # var_test_expand = torch.autograd.Variable( torch.Tensor( [ [1 ,2 ,3 ,4 , 5, 6] , [7,8,9,10,11,12] , [ 13,14,15,16,17,18 ] ] ) )
    # var_test_expanded = var_test_expand.expand( 2, 3, 6 ) # expand the (3 , 6) into higher dimensions
    # print(var_test_expanded)

    # var_test_mult = torch.autograd.Variable( torch.Tensor( [ [ 1 ] , [ 2 ] ] ) )
    # var_test_fac = torch.autograd.Variable( torch.Tensor( [ 2 ] )  )
    # var_test_mult = var_test_mult.mul( var_test_fac )
    # print( var_test_mult )
    # var_test_mult = var_test_mult * 2 + 10
    # print( var_test_mult )

    var_input = torch.autograd.Variable( torch.Tensor( 64, 4, 8, 20*300 + 17 ) )

    att_model_test = TopicalAttentionGRU(
            param_word_embed_size = 300,# 300 in our model
            param_topic_embed_size = 17, #it depends on the setting
            param_fix_sentence_length = 20, # depends on the setting
            param_rnn_tweet_hidden_size = 75, # traditionally the input_size/4
            param_attention_size = (17 + 75) // 100 * 100 + (17 + 75) % 100,
            param_user_self_tweets = 4, # setting
            param_neighbor_tweets = 7, #in the setting 
            param_constant_tweet_weight_a = 0.5,
            param_rnn_hidden_size = 80,
            param_class_count = 3,
            param_on_cuda = False)


    res=att_model_test( var_input )
    # att_model_test( var_input )

    print( res[ 0 ] )
    print( torch.exp( res[ 0 ] ) )