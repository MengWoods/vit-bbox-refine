# This command finds all label ID mask files in the 'train' split and copies them.
find ~/Downloads/Dataset/cityscapes/gtFine_trainvaltest/gtFine/train -name "*_gtFine_labelIds.png" -exec cp {} ~/github/vit-bbox-refine/src/data/train_masks/ \;

echo "Finished copying training masks."