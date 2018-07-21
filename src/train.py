import torch #torch is a file named torch.py
from torch.autograd import Variable # torch here is a folder named torch
from torchnet import meter
from models import topicalAttentionGRU #this is filename, once imported, you can use the classes in it
# equal to from models.topicalAttentionGRU import TopicalAttentionGRU

from settings import *
from utils import *

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2" #cuda device id

# loss_func=nn.CrossEntropyLoss()

def train(**kwargs):
    '''
    begin training the model
    *kwargs: train(1,2,3,4,5)=>kwargs[0] = 1 kwargs[1] = 2 ..., kwargs is principally a tuple
    **kwargs: train(a=1,b=2,c=3,d=4)CustomPreProcessor=>kwargs[a] = 1, kwargs[b] = 2, kwargs[c] = 3, kwargs[d] = 4, kwargs is principally a dict
    function containing kwargs *kwargs **kwargs must be written as: def train(args,*args,**args)
    '''

    saveid = latest_save_num() + 1
    save_path = '%s/%d' % ( SAVE_DIR , saveid ) #the save_path is 
    print("logger save path: %s"%(save_path) )
    if not os.path.exists( save_path ):
        os.makedirs( save_path )
    log_path_each_save = '%s/log.txt' % save_path
    model_path_each_save = '%s/model' % save_path
    logger = get_logger(log_path_each_save)


    config = DefaultConfig()
    config.set_attrs(kwargs) # settings here, avalid_data_utillso about whether on cuda
    # print(config.get_attrs())
    epochs = config.epochs
    batch_size = config.batch_size

    if config.on_cuda: # determine whether to run on cuda        
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            logger.info('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    if config.model == 'TopicalAttentionGRU':
        model = topicalAttentionGRU.TopicalAttentionGRU(
            param_word_embed_size = WORD_EMBEDDING_DIMENSION,# 300 in our model
            param_topic_embed_size = TOPIC_EMBEDDING_DIMENSION, #it depends on the setting
            param_fix_sentence_length = TWITTER_LENGTH, # depends on the setting
            param_rnn_tweet_hidden_size = WORD_EMBEDDING_DIMENSION//4, # the size of tweet level rnn
            param_attention_size = (TOPIC_EMBEDDING_DIMENSION + WORD_EMBEDDING_DIMENSION//4 ) // 64 * 64 + (TOPIC_EMBEDDING_DIMENSION + WORD_EMBEDDING_DIMENSION//4 ) % 64,
            param_user_self_tweets = USER_SELF_TWEETS, # setting
            param_neighbor_tweets = NEIGHBOR_TWEETS, #in the setting 
            param_constant_tweet_weight_a = CONST_TWEET_WEIGHT_A,
            param_rnn_hidden_size = ( WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION) // 4,
            param_class_count = TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT,
            param_on_cuda = config.on_cuda)

    if config.on_cuda:
        logger.info('Model run on GPU')
        model = model.cuda()
        logger.info('Model initialized on GPU')
    else:
        logger.info('Model run on CPU')
        model = model.cpu()
        logger.info('Model initialized on CPU')


    #print('logger-setted',file=sys.stderr)
    logger.info( model.modelname ) #output the string informetion to the log
    logger.info( str( config.get_attrs() ) ) #output the string information to the log

    #read in the trainset and the trial set
    train_data_manager = DataManager( batch_size , TRAINING_INSTANCES ) #Train Set
    train_data_manager.load_dataframe_from_file( TRAIN_SET_PATH )
    #valid_data_manager = DataManager( batch_size , TESTING_INSTANCES) #TestSet
    #valid_data_manager.load( TEST_SET_PATH )
    
    #set the optimizer parameter, such as learning rate and weight_decay, function Adam, a method for Stochastic Optizimism
    lr = config.learning_rate#load the learning rate in config, that is settings.py
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=config.weight_decay #weight decay that is L2 penalty that is L2 regularization, usually added after a cost function(loss function), for example C=C_0+penalty, QuanZhongShuaiJian, to avoid overfitting
                                )
    # By default, the losses are averaged over observations for each minibatch. 
    # However, if the field size_average is set to False, the losses are instead 
    # summed for each minibatch
    
    criterion = torch.nn.KLDivLoss( size_average = False ) # input is log probabilities, while target not
    #criterion = torch.nn.CrossEntropyLoss( size_average = False )#The CrossEntropyLoss, My selector in my notebook = loss + selecting strategy(often is selecting the least loss)
    
    #once you have the loss function, you also have to train the parameters in g(x), which will be used for prediction
    loss_meter = meter.AverageValueMeter() #the loss calculated after the smooth method, that is L2 penalty mentioned in torch.optim.Adam
    confusion_matrix = meter.ConfusionMeter( SENTIMENT_COUNT ) #get confusionMatrix, the confusion matrix is the one show as follows:
    '''                    class1 predicted class2 predicted class3 predicted
    class1 ground tr uth  [[4,               1,               1]
    class2 ground truth   [2,               3,               1]
    class2 ground truth   [1,               2,               9]]
    '''
    model.train()
    pre_loss = 1e100
    best_acc = 0
    for epoch in range( epochs ):
        '''
        an epoch, that is, train data of all barches(all the data) for one time
        '''

        #the chunk has to be initialized after an epoch, have already shifted to previous place since the whole frame needs only loaded once without chunk
        #tramodelin_data_manager.load_dataframe_from_file( TRAIN_SET_PATH )

        loss_meter.reset()
        confusion_matrix.reset()

        # I think better needs a reshuffle in a chunk when chunkized

        train_data_manager.reshuffle_dataframe()

        n_batch = train_data_manager.n_batches() # it was ceiled, so it is "instances/batch_size + 1"

        batch_index = 0
        for batch_index in range( 0 , n_batch - 1):
            ( x , y ) = train_data_manager.next_batch() # this operation is time consuming

            x = Variable( torch.from_numpy( x ).float(  ) )
            #y = Variable( torch.LongTensor( y ) , requires_grad = False )
            y = Variable( torch.FloatTensor( y ) , requires_grad = False )
            
            logger.info('Begin fetching a batch')
            loss , scores , corrects = eval_batch( model , x , y , criterion , config.on_cuda , config.batch_size)
            logger.info('End fetching a batch, begin optimizer')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info('End optimizer')
            #function loss_meter.add:
            #Computes automatically the averaged loss of all the input losses
            #loss the Variable,
            #loss.data the (presumably size 1) Tensor, that has 1 dimension, so you have to use data[0] to access the first value at data
            #loss.data[0] the (python) float at position 0 in the tensor
            loss_meter.add( loss.data[ 0 ] )
            #function confusion_matrix.add:
            #Computes automatically the confusion matrix of K x K size where K is no of classes
            #Args:
            #    predicted (tensor): Can be an N x K tensor of predicted scores obtained from
            #        the model for N examples and K classes or an N-tensor of
            #        integer values between 0 and K-1.
            #    target (tensor): Can be a N-tensor of integer values assumed to be integer
            #        values between 0 and K-1 or N x K tensor, where targets are
            #        assumed to be provided as one-hot vectors
            
            predition_scores = torch.sum( scores.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )
            prediction_y = torch.sum( y.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )

            confusion_matrix.add( predition_scores.data , prediction_y.data )

            if ( batch_index + 1 ) % 5 == 0:# if batch_index == 10 then display the accuracy of the batch
                #==========Cuda 9.0 updates to Cuda 7.5: corrects
                #          was originally an array, now it is a tensor,
                #          so please convert it to numpy getting rid of accuracy=0.000
                #print( corrects )
                corrects = corrects.float()
                #==========Cuda 9.0 updates to Cuda 7.5: corrects
                #          was originally an array, now it is a tensor,
                #          so please convert it to numpy getting rid of accuracy=0.000
                
                accuracy = 1.0 * corrects / config.batch_size
                logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f\taccuracy: %f' % ( epoch , epochs , batch_index , n_batch , loss_meter.value()[0] , accuracy ) ) # .value()[0] is the loss value

        # 128 % 128 = 0, 128 / 128 = 1, if TRAINING_INSTANCES % batch_size == 0 then the last batch size is 0
        if TRAINING_INSTANCES % batch_size == 0:
            train_data_manager.set_current_cursor_in_dataframe_zero()
        else:
            batch_index += 1 # the value can be inherented
            ( x , y ) = train_data_manager.tail_batch()
            x = Variable( torch.from_numpy( x ).float() )
            # y = Variable( torch.LongTensor( y ) , requires_grad = False )
            y = Variable( torch.FloatTensor( y ) , requires_grad = False )

            # score = its = probabilities of [ 0 ~ 9 * 3]
            loss , scores , corrects = eval_batch( model , x , y , criterion , config.on_cuda , config.batch_size) # scores: the softmax, already calculated from the logsoftmax
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.add( loss.data[ 0 ] )

            # prediction = probabilities of [ 0, 1, 2]
            predition_scores = torch.sum( scores.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )
            
            prediction_y = torch.sum( y.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )

            confusion_matrix.add( predition_scores.data , prediction_y.data )

            if ( batch_index + 1 ) % 5 == 0:# if batch_index == 10 then display the accuracy of the batch
                #==========Cuda 9.0 updates to Cuda 7.5: corrects
                #          was originally an array, now it is a tensor,
                #          so please convert it to numpy getting rid of accuracy=0.000
                #print( corrects )
                corrects = corrects.float()
                #==========Cuda 9.0 updates to Cuda 7.5: corrects
                #          was originally an array, now it is a tensor,
                #          so please convert it to numpy getting rid of accuracy=0.000

                accuracy = 1.0 * corrects / config.batch_size
                logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f\taccuracy: %f' % ( epoch , epochs , batch_index , n_batch , loss_meter.value()[0] , accuracy ) ) # .value()[0] is the loss value  y = Variable( torch.LongTensor( y ) , requires_grad = False )
            
            # loss , scores , corrects = eval_batch( model , x , y , criterion , config.on_cuda )
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # loss_meter.add( loss.data[ 0 ] )
            # x = Variable( torch.from_numpy( x ).float() )
            # y = Variable( torch.LongTensor( y ) , requires_grad = False )
            
            # confusion_matrix.add( scores.data , y.data ) 
            # if ( batch_index + 1 ) % 5 == 0:# if batch_index == 10 then display the accuracy of the batch
            #     accuracy = 1.0 * corrects / config.batch_size
            #     logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f\taccuracy: %f' % ( epoch , epochs , batch_index , n_batch , loss_meter.value()[0] , accuracy ) ) # .value()[0] is the loss value

        # after an epoch it should be evaluated
        model.eval() # switch to evaluate model
        #if ( batch_epochsindex + 1 ) % 25 == 0:# every 50 batches peek its accuracy and get the best accuracy
        confusion_matrix_value = confusion_matrix.value()
        acc = 0
        for i in range(SENTIMENT_COUNT):
            acc += confusion_matrix_value[i][i] #correct prediction count
        acc = acc / confusion_matrix_value.sum() #the accuracy, overall accuracy in an epoch
        the_overall_averaged_loss_in_epoch = loss_meter.value()[0] # a 1-dim tensor with lenth 1, so you have to access the element by [0]
        logger.info( 'epoch: %d/%d\taverage_loss: %f\taccuracy: %f' % ( epoch, epochs, the_overall_averaged_loss_in_epoch, acc ) )
        model.train() # switch to train model

        #if accuracy increased, then save the model and change the learning rate
        if acc > best_acc:
            #save the model
            
            #==========model do not save full
            model.save(model_path_each_save)
            #model.save_full(model_path_each_save)
            #torch.save( model, model_path_each_save )
            #==========model do not save full
            
            logger.info('model saved to %s'%model_path_each_save)
            #change the learning rate
            if epoch <= 5:
                lr = lr * config.lr_decay
            else:
                if epoch <= 10:
                    lr = lr * ( config.lr_decay +  0.03 * config.lr_decay)
                else:
                    lr = lr * ( config.lr_decay + 0.05 * config.lr_decay)

            logger.info( 'learning_rate changed to %f'% lr )
            for param_group in optimizer.param_groups:
                param_group['lr']=lr

            best_acc = acc

        pre_loss=loss_meter.value()[0]


def eval_batch(model,x,y,criterion,on_cuda, batch_size):
    '''
    evaluate the logits of each instance, loss, corrects in a batch
    '''
    if on_cuda:
        x , y = x.cuda() , y.cuda()
    else:
        x , y = x.cpu() , y.cpu()

    logits = model(x) # batch_size * dim
    # logits are the probabilities of the logsoftmax, should be converted to probabilities of softmax

    # since the size_average parameter==False, the loss is the sumed loss of the batch. The loss is a value rather than a vector

    loss = criterion( logits , y ) # CrossEntropyLoss takes in a vector and a class num ( usually a index num of the vector )
    
    its = torch.exp( logits )
    predition_its = torch.sum( its.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )
    model_training_predicts = torch.max( predition_its, 1)[ 1 ]
    # model_training_predicts = torch.max( logits , 1)[ 1 ] # [0] : max value of dim 1 [1]: max index of dim 1 LongTensor
    
    # model_training_predicts = torch.fmod( model_training_predicts, SENTIMENT_COUNT )
    # prediction_y : probabilitie of [ 0 1 2]
    # predicts: index of prediction
    predition_y = torch.sum( y.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )
    y_predicts = torch.max( predition_y, 1)[ 1 ] # the index instead of the value

    assert model_training_predicts.size() == y_predicts.size()

    #==========data: Variable to tensor
    corrects = ( model_training_predicts.data == y_predicts.data ).sum( )
    
    return loss , its , corrects

def eval(model,data_manager,criterion,on_cuda):
    '''
    evaluate the accuracy of all epochs
    currently unused, a good example
    '''
    model.eval()#Sets the module in evaluation mode. refer to the pytorch nn manual
    confusion_matrix=meter.ConfusionMeter(CLASS_COUNT)
    loss_meter=meter.AverageValueMeter()
    for i,(x,y) in enumerate(data_manager.next_batch()):
        x=Variable(torch.from_numpy(x).float(logger),volatile=True)
        y=Variable(torch.LongTensor(y),volatile=True)
        loss,scores,corrects=eval_batch(model,x,y,criterion,on_cuda)
        loss_meter.add(loss.data[0])
        confusion_matrix.add(scores.data,y.data)
    acc=0
    cmvalue=confusion_matrix.value()
    for i in range(CLASS_COUNT):
        acc+=cmvalue[i][i]
    acc/=cmvalue.sum()
    model.train()
    return loss_meter.value()[0],acc

def test(model):
    pass

def predict( model_dir, mtype='TopicalAttentionGRU'):
    '''
    load the model and conduct the prediction
    '''

    model_path = '%s/model'%model_dir
    output_path = '%s/res.txt'%model_dir

    config = DefaultConfig() # Just take the default config to do the prediction work

    if mtype=='TopicalAttentionGRU':
        model=topicalAttentionGRU.TopicalAttentionGRU(
            param_word_embed_size = WORD_EMBEDDING_DIMENSION,# 300 in our model
            param_topic_embed_size = TOPIC_EMBEDDING_DIMENSION, #it depends on the setting
            param_fix_sentence_length = TWITTER_LENGTH, # depends on the setting
            param_rnn_tweet_hidden_size = WORD_EMBEDDING_DIMENSION//4, # the size of tweet level rnn
            param_attention_size = (TOPIC_EMBEDDING_DIMENSION + WORD_EMBEDDING_DIMENSION//4 ) // 64 * 64 + (TOPIC_EMBEDDING_DIMENSION + WORD_EMBEDDING_DIMENSION//4 ) % 64,
            param_user_self_tweets = USER_SELF_TWEETS, # setting
            param_neighbor_tweets = NEIGHBOR_TWEETS, #in the setting 
            param_constant_tweet_weight_a = CONST_TWEET_WEIGHT_A,
            param_rnn_hidden_size = ( WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION) // 4,
            param_class_count = TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT,
            param_on_cuda = config.on_cuda)
    print('Loading trained model')
    model.load( model_path )

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            print('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    if config.on_cuda:
        model=model.cuda()
    else:
        model=model.cpu()
    
    # print(model)
    print('Begin loading data')
    datamanager=DataManager( param_batch_size = config.batch_size, param_training_instances_size = TESTING_INSTANCES) # the batch_size makes no differences
    datamanager.load_dataframe_from_file( TEST_SET_PATH )
    n_batch = datamanager.n_batches()
    res=[]

    batch_index = 0

    for batch_index in range(n_batch - 1):
        ( x , y ) = datamanager.next_batch()
        x = Variable( torch.from_numpy(x).float(), requires_grad=False)
        x = x.cuda()
        logits = model.forward(x)

        # predition_scores = torch.sum( scores.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )
        # #model_training_predicts = torch.max( predition_scores, 1)[ 1 ]

        # _ , predict = torch.max(predition_scores, 1) # predict is the first dimension , its the same as [ 1 ] 
        # # print(predict)
        # res.extend( predict.data ) # model predicts
        scores = torch.exp( logits )
        # print( scores.data )

        res.extend( scores.data.cpu().numpy() ) # if using CPU then res.data.numpy()

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        ( x , y ) = datamanager.tail_batch()
        x = Variable( torch.from_numpy(x).float(), requires_grad=False)
        x = x.cuda()
        logits = model.forward( x )
        scores = torch.exp( logits )
        # print( scores.data )
        # predition_scores = torch.sum( scores.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )

        # _ , predict = torch.max(predition_scores, 1) # predict is the first dimension , its the same as [ 1 ] 
        # # print(predict)
        # res.extend( predict.data ) # model predicts

        res.extend( scores.data.cpu().numpy() ) # if using CPU then res.data.numpy() else res.data.cpu().numpy

        print( '%d/%d'%( batch_index, n_batch ) )  
    # print('Hello')

    res = res[ :TESTING_INSTANCES ]
    
    # print( res )

    numpy.savetxt( output_path, res, fmt='%f') # no %d

def calculate_accuracy(model_dir):
    fpInTestSet = open(TEST_SET_PATH, 'rt')
    fpInPredicted = open('%s/res.txt'%model_dir,'rt')
    rightPredictionCount = 0
    
    for alineG in fpInTestSet:
        grtDisrtibution = alineG.strip().split( ' ', TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT )[ 0: TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT ]
        grtArray = numpy.array( grtDisrtibution ).astype( numpy.float32 )
        grtArraySentiment = numpy.sum( numpy.reshape( grtArray, ( TOPIC_SENTIMENT_COUNT , SENTIMENT_COUNT ) ) ,axis = 0 )
        grtIndex = numpy.argmax( grtArraySentiment, axis = None )
        alineP = fpInPredicted.readline()
        predictedDistribution = alineP.strip().split( ' ' )
        assert len(grtDisrtibution) == len(predictedDistribution)
        predictedArray = numpy.array( predictedDistribution).astype(numpy.float32 )
        predictedArraySentiment = numpy.sum( numpy.reshape( predictedArray, ( TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ) ), axis = 0 )
        predictedIndex = numpy.argmax( predictedArraySentiment, axis = None )
        if grtIndex == predictedIndex:
            rightPredictionCount += 1
    fpInTestSet.close()
    fpInPredicted.close()
    
    acc = rightPredictionCount / TESTING_INSTANCES
    print(acc)

if __name__=='__main__':
    # var1 = torch.autograd.Variable(torch.Tensor( [[ 0.01,0.02,0.015],[ 0.02, 0.01, 0.015],[ 0.01, 0.015, 0.02]] ) )
    # max_output = torch.max( var1 , 1)[ 1 ]
    # var2 = torch.autograd.Variable(torch.Tensor( [5,6,7]))
    # assert max_output.size() == var2.size()
    


    #train(  )
    
    predict('%s/0'%SAVE_DIR) # best: 22
    calculate_accuracy('%s/0'%SAVE_DIR)