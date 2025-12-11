find ~/Downloads/Dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test -name "*_leftImg8bit.png" -exec cp {} ~/github/vit-bbox-refine/src/data/test_images/ \;
echo "Finished copying testing images."
find ~/Downloads/Dataset/cityscapes/gtFine_trainvaltest/gtFine/test -name "*_gtFine_labelIds.png" -exec cp {} ~/github/vit-bbox-refine/src/data/test_masks/ \;
echo "Finished copying testing masks."

ls ../src/data/test_images | wc -l
ls ../src/data/test_masks | wc -l
