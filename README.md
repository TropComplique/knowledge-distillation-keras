# Knowledge distillation with Keras

Keras implementation of Hinton's knowledge distillation (KD), a way of transferring knowledge from a large model into a smaller model.

## Summary
* I use Caltech-256 dataset for a demonstration of the technique.
* I transfer knowledge from Xception to MobileNet-0.25 and SqueezeNet v1.1.
* Results: 

| model | accuracy, % | top 5 accuracy, %| logloss |
| --- | --- | --- | --- | 
| Xception    | 82.3 | 94.7 | 0.705 |
| MobileNet-0.25 | 64.6 | 85.9 | 1.455 |
| MobileNet-0.25 with KD | 66.2 | 86.7 | 1.464 |
| SqueezeNet v1.1 | 67.2 | 86.5 | 1.555 |
| SqueezeNet v1.1 with KD | 68.9 | 87.4 | 1.297 |


## Implementation details
* I use pretrained on ImageNet models.
* For validation I use 20 images from each category.
* For training I use 100 images from each category.
* I use random crops and color augmentation to balance the dataset.
* I resize all images to 299x299.
* In all models I train the last two layers.

## Notes on `flow_from_directory`
I use three slightly different versions of Keras' `ImageDataGenerator.flow_from_directory`:
* original version for initial training of Xception and MobileNet.
* ver1 for getting logits from Xception. Now `DirectoryIterator.next` also outputs image names.
* ver2 for knowledge transfer. Here `DirectoryIterator.next` packs logits with hard true targets.
All three versions only differ in `DirectoryIterator.next` function.

## Requirements
* Python 3.5
* Keras 2.0.6
* torchvision, Pillow
* numpy, pandas, tqdm

## References
[1] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
