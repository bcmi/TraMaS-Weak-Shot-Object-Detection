import os
import json
import torch
def extract_features(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
):
    output_folder = os.path.join(cfg.OUTPUT_DIR, 'simtrain')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    features=[]
    labels=[]
    i=0
    with torch.no_grad():
        for images,targets,_ in data_loader:
            i+=1
            print(i)
            images=images.to(device)
            targets=[target.to(device) for target in targets]
            feature,label=model(images,targets)
            features.append(feature.cpu())
            labels.append(label.cpu())
        features=torch.cat(features)
        labels=torch.cat(labels)
        print(features.shape)
        feature_path=os.path.join(output_folder,'features.json')
        label_path=os.path.join(output_folder,'labels.json')
        # json.dump(features,open(feature_path,'w'))
        # json.dump(labels,open(label_path,'w'))

        
