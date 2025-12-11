# Assuming your downloaded cityscapes root is in a folder named 'cityscapes_raw'
# This command finds all image files in the 'train' split and copies them to your target folder.
find ~/Downloads/Dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train -name "*_leftImg8bit.png" -exec cp {} ~/github/vit-bbox-refine/src/data/train_images/ \;

echo "Finished copying training images."

# This command finds all label ID mask files in the 'train' split and copies them.
find ~/Downloads/Dataset/cityscapes/gtFine_trainvaltest/gtFine/train -name "*_gtFine_labelIds.png" -exec cp {} ~/github/vit-bbox-refine/src/data/train_masks/ \;

echo "Finished copying training masks."


ls ../src/data/train_images | wc -l
ls ../src/data/train_masks | wc -l
