import timm
from subnetworks import submasking

# Grab pretrained model
model = timm.create_model('resnet50', pretrained=True)

# Print a summary of all the weights, this is usefull to know how to set up the parameter selection function below
weight_summary = ""
for name, param in model.named_parameters():
    row = f"{name}: {param.shape}, {param.numel()} elements, requires_grad={param.requires_grad}\n"
    weight_summary += row


# Select which parameters to train, mask or freeze based on the name of the parameter.
def par_sel(name, param):
    if not param.requires_grad:
        return 'freeze'
    for l in ['conv', 'downsample']:
        if l in name:
            return 'mask', name
    if 'fc' in name:
        return 'mask', name  # Replace 'mask' here with 'train' if you don't want to mask the fc layer
    return 'freeze'


# Initialize with a normal distribution of mean one and std zero, i.e. initialize every score to a 1.0
scores_init = submasking.normal_scores_init(1.0, 0.0)

# Create a masked version of the model, using the default settings of a threshold of 0
model = submasking.SubmaskedModel(model, parameter_selection=par_sel, scores_init=scores_init, shell_mode='replace')

# ... train the model