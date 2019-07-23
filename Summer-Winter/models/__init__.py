def create_model(opt):
    print(opt.model)
    if opt.model == 'DIP':
        from .DIP_model import DIPModel
        model = DIPModel()
    elif opt.model == 'DIP_AE':
        from .DIP_autoencoder_model import DIP_AE_Model
        model = DIP_AE_Model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
