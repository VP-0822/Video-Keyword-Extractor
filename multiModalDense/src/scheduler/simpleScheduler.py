try:
    import torch_xla.core.xla_model as xm
except ImportError:
    print('[WARNING]: Not able to import "torch_xla.core.xla_model", try pip install..')    
class SimpleScheduler(object):
    """
        Simple Learning Rate scheduler for optimized learning. LR Scheduler's generally change LR to lower numbers as we approach local minima 
    """
    
    def __init__(self, optimizer, learning_rate, device):
        """
            Args:
                optimizer: Gradient update optimizer. Generally Adam Optimizer is preferred.
                learning_rate: Learning rate number
                device: Device used for training
        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.device = device
        
    def step(self):
        """
            Takes optimizer step
        """
        self.optimizer.param_groups[0]['lr'] = self.get_lr()
        if self.device.type == 'xla':
            xm.optimizer_step(self.optimizer)
            xm.mark_step()
        else:
            self.optimizer.step()
    
    def get_lr(self):
        """
            return learning rate
        """
        return self.learning_rate