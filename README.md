# Image Classification using ResNet, ResNeXt, and DenseNet

### Training of Models
1. BasicNet
Use this command to train the BasicNet:
`python Train.py --ModelNumber 1`
2. BatchNormNet
Use this command to train the BatchNormNet:
`python Train.py --ModelNumber 2`
3. ResNet
Use this command to train the ResNet:
`python Train.py --ModelNumber 3`
4. ResNeXt
Use this command to train the ResNeXt:
`python Train.py --ModelNumber 4`
5. DenseNet
Use this command to train the DenseNet:
`python Train.py --ModelNumber 5`

### Testing of Models
1. BasicNet
Use this command to test the BasicNet:
`python Test.py --ModelPath ./Checkpoints/BasicNet/14model.ckpt`
2. BatchNormNet
Use this command to test the BatchNormNet:
`python Test.py --ModelPath ./Checkpoints/BatchNormNet/14model.ckpt`
3. ResNet
Use this command to test the ResNet:
`python Test.py --ModelPath ./Checkpoints/ResNet/14model.ckpt`
4. ResNeXt
Use this command to test the ResNeXt:
`python Test.py --ModelPath ./Checkpoints/ResNeXt/14model.ckpt`
5. DenseNet
Use this command to test the DenseNet:
`python Test.py --ModelPath ./Checkpoints/DenseNet/14model.ckpt`
