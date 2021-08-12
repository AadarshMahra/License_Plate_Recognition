# License Plate Recognition
This project locates license plates within larger images. Then, the license plate is read using a Keras model that is pre-trained on a Kaggle [dataset](https://www.kaggle.com/aladdinss/license-plate-digits-classification-dataset). 

 
![bmw](https://github.com/AadarshMahra/License_Plate_Recognition/blob/main/media/bmw.jpg?raw=true)
## Locates Image
![located](https://media.discordapp.net/attachments/699093898915610694/795577611854151680/Screen_Shot_2021-01-04_at_1.00.01_AM.png?width=800&height=462)

## Image is cropped. Then it's read using the Keras model  
![read](https://media.discordapp.net/attachments/699093898915610694/875283469192794122/new_guess.png)


## FUTURE STEPS: 
### For the model : 
- test model on more out of sample data 
- create confusion matrix to further examine model accuracy 

### For the rest of the project: 
- improve accuracy for all images, not just clear ones
- avoid mistake where two contours are found in a zero 
![edge](https://media.discordapp.net/attachments/699093898915610694/875287868824780851/zero_double_contour.png?width=379&amp;height=600" )
