# Cat-Face-Recognition Replica

Replica for Research -Original Creator: Taehee Brad Lee -  - https://github.com/kairess )

- Youtube:   https://www.youtube.com/watch?v=7_97SlAigPs&t=210s

- Blog:  https://blog.naver.com/dnjswns2280/221886031616


- Introduction : 

	- Overlapping image(glasses) on base image(cat)
	- Two data models used
		- 1. Cat face recognition model
		- 2. Finding the eyes, nose, ears, and mouth of the cat image
   
   
  Directory Structure:
    	- cat-dataset: 8000 cat images and 8000 landmarks for each cat image face
	- dataset: cat-dataset training preprocess images
	- images: glasses image
	- logs: model train check point save file
	- models: Trained dataset with cat face and landmark
	- result: Result images with cats with glasses
	- sample: images before glasses are applied

- py.files 
	- 1. helper.py:  image size function, that improves glasses application  on the cat images
	- 2. main.py: Basic image start file to display images
	- 3. preprocess_lmks.py:  Saves preprocessed  the cat face landmark images inside the dataset 
	- 4. preprocessing.py: Saves cropped cat face images in the dataset
	- 5. train.py: File that trains model to crop cat face from cat image
	- 6. train_lmks.py: File that trains model to find cat landmark 
	- 7. test.py: Displays the final trained models results
  ![catface1](https://user-images.githubusercontent.com/25238652/126759714-9cc85d65-9cf1-46c5-90b8-1c6ba9a10d22.PNG)
![catface5](https://user-images.githubusercontent.com/25238652/126759720-14598a9f-de8b-4a8e-836f-8dc433d4143b.PNG)
![catface9](https://user-images.githubusercontent.com/25238652/126759732-d4bf9288-b377-4424-b11d-d1e8b25fc9be.PNG)
![catface16](https://user-images.githubusercontent.com/25238652/126759744-9318bb88-641d-4ae0-8b91-cb7f0cae3d27.PNG)
![catface17](https://user-images.githubusercontent.com/25238652/126759748-bf829166-250e-45fe-ab94-f3da01db6e34.PNG)
![catface7](https://user-images.githubusercontent.com/25238652/126759752-1353640b-49a2-4633-8626-d60005212046.PNG)
![catface6](https://user-images.githubusercontent.com/25238652/126759756-9deb6610-b9c9-4155-ba79-030dbbbc3e18.PNG)

  
