# -Sign-Language-Fingerspelling-Assistant-ASL-Alphabet-


# The data link

https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

# Description
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.


The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
These 3 classes are very helpful in real-time applications, and classification.
The test data set contains a mere 29 images, to encourage the use of real-world test images


# Models

1- CNN with 30 epoch got accuracy: 0.9670 on Training / val_accuracy: 0.9734 on Validation

<img width="1432" height="67" alt="image" src="https://github.com/user-attachments/assets/2c01df2a-d463-4047-b475-67f7e3d9f2e8" />



2- EfficientNetB0 with early stopping stopped at 19 epochs and got accuracy : 0.9325 on Training / val_accuracy: 0.9672 on Validation 

<img width="1409" height="55" alt="image" src="https://github.com/user-attachments/assets/d8d4a6f5-387d-4d61-a778-1f0e21ba0039" />



3- EfficientNEtB0_V2 with early stopping stopped at 19 epochs and got accuracy : 0.9925 on Training / val_accuracy: 0.9872 on Validation

![WhatsApp Image 2025-12-22 at 3 30 14 AM](https://github.com/user-attachments/assets/7b2d4604-b605-4f0a-a865-1fe25b94d4e2)


4- ResNet50   with  10 epoch  and got  accuracy: 0.9827 - val_accuracy: 0.9920

<img width="952" height="44" alt="image" src="https://github.com/user-attachments/assets/a5c9f177-96ed-4e22-9371-47f14b0fa1e8" />





# Deployment

We used streamlit to provide us with a simple GUI 























# Models Google Drive Links 
Link : https://drive.google.com/drive/folders/1TF81jTd-MxCYPQrp-ac_d0ZJz9f7Qfnb?usp=sharing
