from keras import callbacks

class BasicModelCallback(callbacks.Callback):
    def __init__(self, final_model, folderName):
        self.final_model = final_model
        self.folderName = folderName
        pass
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch %d End " % epoch)
        loss = logs['loss']
        # acc  = logs['acc']
        #valloss = logs['val_loss']
        #valacc  = logs['val_acc']
        print('Epoch Training loss: ' + str(loss))
        # print('Epoch Training Accuracy: ' + str(acc))
        print('=========================================')
        if epoch % 20 is 0 and epoch is not 0:
            print('writing to model weights file')
            self.final_model.save_weights(self.folderName + 'trainedModel_' + str(epoch) + '.hdf5')
    
    def on_batch_end(self, batch, logs={}):
        print("Batch %d ends" % batch)
        loss = logs['loss']
        # acc  = logs['acc']
        print('Batch Training loss: ' + str(loss))
        # print('Batch Training Accuracy: ' + str(acc))