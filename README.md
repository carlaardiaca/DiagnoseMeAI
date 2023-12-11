# DiagnoseMe: Predictive Symptom2Disease Analysis
Welcome to the DiagnoseMe repository, your gateway to predictive health analysis. This project focuses on the correlation between symptoms and diseases, offering data-driven insights for informed healthcare decisions.

## Overview
DiagnoseMe leverages state-of-the-art natural language processing and machine learning techniques to predict diseases from textual symptom descriptions. It aims to provide a user-friendly interface for users to input their symptoms and receive a list of possible conditions, helping to bridge the gap between preliminary self-diagnosis and professional medical advice.

## Project Structure
The project is organized into the following folders and files:

- `data/`: Contains the datasets used for training and evaluating the models.
- `gui/`: Source code for the graphical user interface built with Tkinter.
- `results/`: Generated results and graphs from data processing and model evaluation.
- `clean_data.py`: Script for data cleaning and normalization.
- `data_preprocessing.py`: Data preparation including tokenization and encoding.
- `main.py`: The main entry point of the program to run the system.
- `model_selection.py`: Script for selecting the best model using GridSearchCV.
- `model_testing.py`: Model evaluation and performance metrics generation.
- `requirements.txt`: Necessary dependencies to run the project.
- `DiagnoseMe_Predictive_Symptom2Disease_Analysis.ipynb`: This Jupyter notebook contains the detailed workflow of the project, including data cleaning, feature engineering, model selection and optimization and results analysis. 
- `DiagnoseMe_Symptom2Disease_paper.pdf`: A comprehensive report detailing the research methodology, algorithms used, system design, results, and conclusions of the Symptom to Disease prediction project.

## Environment Setup
To install all required dependencies, run the following command:
          pip install -r requirements.txt

## Usage
To launch and run the program and start the disease prediction process, execute:
          python main.py

To use the interface built with Tkinter, execute:
          python predict_TOP3.py

## Technologies Used
- Python: The primary programming language.
- NLTK: Library for natural language processing.
- Scikit-learn: Machine learning tools.
- Pandas, NumPy: Data manipulation and mathematical computations.
- Matplotlib, Seaborn: Data visualization.

## Contributing
Contributions are welcome. If you'd like to contribute, please fork the repository and propose your changes via a pull request.

## Contact
If you have any questions or comments, feel free to open an issue in this repository.


