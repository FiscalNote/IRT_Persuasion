import torch

class IRTNet(torch.nn.Module):
    
    def __init__(self, D_in, num_users, init_type='uniform', C=1e-4, reg_type='l2'):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        
        self.linear = torch.nn.Linear(D_in, 1, bias=False)
        
        self.users = torch.nn.Embedding(num_users, 1) 
        
        self.C = C
        
        if reg_type not in ['l1', 'l2']:
            raise ValueError(f"Unknown Regularization {reg_type}")
        self.reg_type = 2 if reg_type == 'l2' else 1
        
        if init_type == 'uniform':
            torch.nn.init.uniform_(self.users.weight, -0.5, 0.5)
        elif init_type == 'normal':
            torch.nn.init.normal_(self.users.weight, 0, 1)
        else:
            raise ValueError("Unknown init")

    def forward(self, feats, user_id, labels=None, doc_id=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        diff = self.linear(feats)

        user = self.users(user_id)
                
        prod = diff - user

        y_pred = torch.sigmoid(prod)
        
        if labels is not None:
                        
            loss_fct = torch.nn.BCELoss()

            loss = loss_fct(y_pred.reshape(-1), labels.float())
            
            # Regularize
            loss += (torch.norm(self.linear.weight, self.reg_type) + torch.norm(self.users.weight, self.reg_type)) * self.C

        else:
            loss = None
        
        return loss, y_pred, diff

class IdealNet(torch.nn.Module):
    
    def __init__(self, D_in, num_users, user_dim=1, init_type='uniform', use_popularity=True, C=1e-4, reg_type='l2'):

        super().__init__()

        self.use_popularity = use_popularity
        
        self.polarity = torch.nn.Linear(D_in, user_dim, bias=False)
        self.popularity = torch.nn.Linear(D_in, 1)

        self.users = torch.nn.Embedding(num_users, user_dim) 
        
        self.C = C
        
        if reg_type not in ['l1', 'l2']:
            raise ValueError(f"Unknown Regularization {reg_type}")
        self.reg_type = 2 if reg_type == 'l2' else 1
        
        if init_type == 'uniform':
            torch.nn.init.uniform_(self.users.weight, -0.5, 0.5)
        elif init_type == 'normal':
            torch.nn.init.normal_(self.users.weight, 0, 1)
        else:
            raise ValueError("Unknown init")

    def forward(self, feats, user_id, labels=None, doc_id=None):

        direction = self.polarity(feats)
        
        user = self.users(user_id)
                
        # Need sum for multi-dim
        prod = (direction * user).sum(axis=1).unsqueeze(1)
        popular = self.popularity(feats)
                
        if self.use_popularity:
            final_val = prod + popular
        else:
            final_val = prod

        y_pred = torch.sigmoid(final_val)
        
        if labels is not None:
            
            loss_fct = torch.nn.BCELoss()

            loss = loss_fct(y_pred.reshape(-1), labels.float())
            
            # Regularization
            
            all_norm = torch.norm(self.polarity.weight, self.reg_type) 
            all_norm = all_norm + torch.norm(self.users.weight, self.reg_type).float()
            
            if self.use_popularity:
                all_norm = all_norm + torch.norm(self.popularity.weight, self.reg_type)

                    
            loss += all_norm.float() * self.C
            
        else:
            loss = None
        
        return loss, y_pred, prod, popular