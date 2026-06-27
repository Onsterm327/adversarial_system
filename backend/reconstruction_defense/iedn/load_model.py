import torch
from models import Generator
from PurifyModel import EBMModel, GANModel, GLOWModel, NCSNV2Model, DiffusionModel, CleanModel
def load_model(dataset):
    defense_model = Generator().cuda().eval()

    defense_model.load_state_dict(torch.load(f'{dataset}_iedn.pth'))
    defense_model.eval()

    return GANModel(defense_model)