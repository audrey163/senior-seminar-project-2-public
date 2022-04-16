def save_model(model):
    import datetime
    timestamp = str(datetime.datetime.now())
    PATH = 'models/' + timestamp
    torch.save(model.state_dict(),PATH)

