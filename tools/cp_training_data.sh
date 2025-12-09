# Assuming your downloaded cityscapes root is in a folder named 'cityscapes_raw'
# This command finds all image files in the 'train' split and copies them to your target folder.
find ~/Downloads/Dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train -name "*_leftImg8bit.png" -exec cp {} ~/github/vit-bbox-refine/src/data/train_images/ \;

echo "Finished copying training images."