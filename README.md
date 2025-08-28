# Cat & Dog Image Classification Web App 🐾
An interactive web application built with Streamlit and powered by a TensorFlow CNN model to classify images of cats and dogs.

## 📊 Dataset
The CNN model was trained on the popular Cats and Dogs dataset. You can find the original dataset on Kaggle or other public repositories. A well-curated dataset is crucial for training an accurate model.<br>
[Download Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

# 🚀 Core Features
Interactive Image Upload: Users can easily upload an image file (JPG, PNG, JPEG) through a simple web interface.

Real-time Classification: The app uses a pre-trained Convolutional Neural Network (CNN) to predict whether the uploaded image contains a cat or a dog.

Probability Display: Shows the confidence score of the model's prediction.

User-Friendly Interface: Built with Streamlit for a clean, intuitive, and responsive user experience.

# 🎬 Demo Video 
<img width="880" height="450" src="https://github.com/subhadipsinha722133/Cat-and-Dog-Image-Classification-Project/blob/main/demo.gif" alt="Python project demo">


# 🛠️ Technologies & Libraries Used
This project leverages a powerful stack of Python libraries for machine learning and web deployment:

Streamlit: For creating and sharing the interactive web application.

TensorFlow: The core deep learning framework used to build and train the CNN model.

Pillow: For image manipulation and preprocessing tasks.

OpenCV-Python: Used for advanced image processing and handling.

NumPy: For efficient numerical operations and data manipulation.

# 🧠 Model Architecture
The classification model is a Convolutional Neural Network (CNN) built with TensorFlow. A CNN is a deep learning architecture specifically designed for analyzing visual imagery.

Our model consists of several layers:

Convolutional Layers: To automatically and adaptively learn spatial hierarchies of features from the input images.

Pooling Layers (Max Pooling): To reduce the spatial dimensions of the output volume, which helps in reducing computational power required and controlling overfitting.

Flatten Layer: To convert the 2D feature maps into a 1D vector.

Dense (Fully Connected) Layers: To perform classification based on the features extracted by the convolutional layers.

This architecture allows the model to effectively learn the distinct features of cats and dogs for accurate classification.

# ⚙️ Local Setup and Installation
Follow these steps to get the project up and running on your local machine.

1. Clone the Repository:

git clone https://github.com/subhadipsinha722133/Cat-and-Dog-Image-Classification-Project.git <br>
cd Cat-and-Dog-Image-Classification-Project


2. Create and Activate a Virtual Environment:<br>
It's highly recommended to use a virtual environment to manage project dependencies.<br>

# For Windows
python -m venv venv <br>
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv <br>
source venv/bin/activate

3. Install Dependencies:<br>
Create a requirements.txt file in the root of your project with the following content:<br>

- streamlit
- tensorflow
- opencv-python
- numpy
- pillow

Then, install all the required libraries using pip:<br>

pip install -r requirements.txt<br>

4. Run the Streamlit App:<br>
Assuming your main script is named app.py, run the following command in your terminal:<br>

- streamlit run app.py

Your web browser should automatically open with the application running.

# 📖 How to Use the App
Launch the App: Run the streamlit run app.py command from your project directory.

Upload an Image: Use the file uploader widget on the web page to select an image of a cat or a dog.<br>

View the Prediction: The application will process the image and display the model's prediction (either "Cat" or "Dog") along with the confidence score.


