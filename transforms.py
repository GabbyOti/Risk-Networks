import numpy as np

class Transform:

    def __init__(
            self,
            name="identity_clip",
            **kwargs):   
        '''
        Instantiate the object to implement transforms defined by a name
        
        Args
        ----
        name (string),      extra parameters,
        "identity_clip",    None
        "logit",            None
        "tanh_clip",        lengthscale
        "tanh",             lengthscale
                       
        
        '''
        self.name = name
        if self.name == "tanh_clip":
            self.lengthscale = kwargs.get('lengthscale',1) #1 is a default lengthscale
            assert self.lengthscale > 0.5
        if self.name == "tanh":
            self.lengthscale = kwargs.get('lengthscale',1) #1 is a default lengthscale
    
    def apply_transform(self,x):
        #maps FROM [0,1]
        return {   
            'identity_clip' : lambda x: x,
            'logit'         : lambda x: np.log(np.maximum(x, 1e-9) / np.maximum(1.0 - x, 1e-9)),
            'tanh_clip'     : lambda x: np.arctanh((x-0.5)/self.lengthscale),
            'tanh'          : lambda x: np.arctanh(2*x - 1)*self.lengthscale
        }[self.name](x)

    def apply_inverse_transform(self,x):
        #maps INTO [0,1]
        return {
            'identity_clip' : lambda x: np.clip(x,0,1),
            'logit'         : lambda x: np.exp(x)/(np.exp(x) + 1.0),
            'tanh_clip'     : lambda x: np.clip(np.tanh(x)*self.lengthscale+0.5, 0,1),
            'tanh'          : lambda x: 0.5*(1 + np.tanh(x/self.lengthscale))
        }[self.name](x)


