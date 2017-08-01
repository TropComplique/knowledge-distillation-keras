# Knowledge distillation with Keras

Keras implementation of Hinton's knowledge distillation, a way of transferring knowledge from a large model into a smaller model.

## Summary
* I use Caltech-256 dataset for a demonstration of the technique.
* I transfer knowledge from Xception to MobileNet.
* Results: ???

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
