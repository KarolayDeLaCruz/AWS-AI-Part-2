import json
import argparse
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

# Command line inputs
parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('path_to_image', help='Path image')
parser.add_argument('checkpoint', help='Checkpoint')
parser.add_argument('--top_k', help='Return top K most likely classes')
parser.add_argument('--category_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')
args_input = parser.parse_args()

# Define default inputs
top_k = int(args_input.top_k) if args_input.top_k else 5
category_names = args_input.category_names if args_input.category_names else 'cat_to_name.json'

# Load cat names
print('--------- Load cat names ---------')
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load checkpoint
print('--------- Load checkpoint ---------')
checkpoint = torch.load(args_input.checkpoint)
# Define pretrained network
print('--------- Define pretrained network ---------')
model = models.vgg11(pretrained=True) if checkpoint['arch'] == 'vgg11' \
    else models.vgg16(pretrained=True) if checkpoint['arch'] == 'vgg16' \
    else models.vgg19(pretrained=True) if checkpoint['arch'] == 'vgg19' \
    else models.vgg11(pretrained=True)
print(checkpoint['arch'])
# Parameters are frozen
print('--------- Parameters are frozen ---------')
for param in model.features.parameters():
    param.requires_grad = False

# Classifier
print('--------- Classifier ---------')
Classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                           nn.ReLU(),
                           nn.Dropout(0.4),
                           nn.Linear(checkpoint['hidden_units'], 102),
                           nn.LogSoftmax(dim=1)
                           )

model.classifier = Classifier
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['m_state_dict'])
print(model)


# Process Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    fig = Image.open(image)
    data_transforms_test = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    fig = data_transforms_test(fig)
    # print(type(fig))

    return fig


# Prediction
def predict(image_path, model, topk=5):
    # model.to(device)
    img = process_image(image_path)
    img = img.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args_input.gpu else torch.device("cpu")
    img = img.to(device)

    model.eval()

    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]

    class_to_idx = {model.class_to_idx[vals]: vals for vals in model.class_to_idx}
    classes = [class_to_idx[item] for item in top_class]

    return top_p, classes


print('--------- Prediction ---------')
# img_test = 'flowers/test/100/image_07899.jpg'
probs, classes = predict(args_input.path_to_image, model, top_k)
print('--------- Define flowers name ---------')
classes = [cat_to_name[classe] for classe in classes]

print('--------- Result ---------')
# Print prediction
print(classes)
print(probs)
