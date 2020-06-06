class SimpleScheduler(object):
    """
        Simple Learning Rate scheduler for optimized learning. LR Scheduler's generally change LR to lower numbers as we approach local minima 
    """
    
    def __init__(self, optimizer, learning_rate):
        """
            Args:
                optimizer: Gradient update optimizer. Generally Adam Optimizer is preferred.
                learning_rate: Learning rate number 
        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        
    def step(self):
        """
            Takes optimizer step
        """
        self.optimizer.param_groups[0]['lr'] = self.get_lr()
        self.optimizer.step()
    
    def get_lr(self):
        """
            return learning rate
        """
        return self.learning_rate