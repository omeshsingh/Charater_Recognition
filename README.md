# Last.fm Song Recommendation System

## Overview

This project showcases a song recommendation system using a dataset from Last.fm, which is downloaded via the Kaggle API. The script preprocesses the data using Pandas and employs the Surprise library to build a recommendation system using Singular Value Decomposition (SVD).

## Features

- **Kaggle API Integration**: Downloads the Last.fm dataset directly from Kaggle.
- **Data Preprocessing**: Cleans and preprocesses the dataset to prepare it for analysis.
- **Collaborative Filtering**: Uses the SVD algorithm to recommend songs to users based on their listening history.
- **Recommendation Function**: Provides personalized song recommendations for a given user.

## Setup

### Prerequisites

- Python 3.x
- Kaggle API
- Pandas
- Scikit-Surprise

### Installation

Install the required Python packages using pip:

```bash
pip install kaggle pandas scikit-surprise
```

### Kaggle API Credentials

Ensure you have your `kaggle.json` file downloaded from your Kaggle account and upload it to your working directory.
https://www.kaggle.com/datasets/neferfufi/lastfm is the dataset used here.

## Usage

### Step 1: Kaggle API Authentication and Dataset Download

- Create a `.kaggle` directory in the root to store the API credentials.
- Move the `kaggle.json` file to this directory and set appropriate permissions.
- Authenticate using the Kaggle API and download the dataset.

```python
import shutil
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate and download dataset
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.move('/content/kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 600)

api = KaggleApi()
api.authenticate()
dataset = 'neferfufi/lastfm'
api.dataset_download_files(dataset, path='.', unzip=True)
```

### Step 2: Load and Preprocess the Data

- Load the Last.fm dataset into a Pandas DataFrame.
- Clean the data by removing rows with missing values and selecting relevant columns.

```python
import pandas as pd

# Load the dataset
filename = '/content/userid-timestamp-artid-artname-traid-traname.tsv'
df = pd.read_csv(filename, sep='\t', on_bad_lines='skip')

# Preprocess the data
df = df[['user_id', 'artist_name', 'track_name']].dropna()

# Assign a rating of 1 for each user-track interaction
df['rating'] = 1

# Reduce the dataset to a subset of users for faster processing
subset_users = df['user_id'].unique()[:100]
df_subset = df[df['user_id'].isin(subset_users)]
```

### Step 3: Build the Recommendation Model

- Load the preprocessed data into the Surprise library's dataset format.
- Split the data into training and testing sets.
- Train the SVD algorithm on the training data and evaluate its performance.

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the data into Surprise
reader = Reader(rating_scale=(1, 1))
data = Dataset.load_from_df(df[['user_id', 'track_name', 'rating']], reader)

# Split into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train using SVD
algo = SVD()
algo.fit(trainset)

# Evaluate the model
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

### Step 4: Generate Song Recommendations

- Use the trained SVD model to generate song recommendations for a specific user.

```python
def get_recommendations(user_id, n=10):
    unique_songs = df['track_name'].unique()
    songs_not_rated = [song for song in unique_songs if not any(df[(df['user_id'] == user_id) & (df['track_name'] == song)].shape[0])]
    predictions = [algo.predict(user_id, song) for song in songs_not_rated]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    return [(pred.iid, pred.est) for pred in top_n]

# Example usage
user_id = 'some_user_id'  # Replace with an actual user ID from the dataset
recommendations = get_recommendations(user_id)
for song, rating in recommendations:
    print(f'Song: {song}, Predicted Rating: {rating}')
```

## Limitations

- The current setup uses a small subset of users for faster processing. Expanding the dataset may improve the model's accuracy.
- The recommendation system assumes implicit feedback, assigning a rating of 1 to all user-song interactions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Scikit-Surprise](http://surpriselib.com/) for the collaborative filtering algorithms used in this project.
