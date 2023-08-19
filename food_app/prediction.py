import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Text
from sklearn.cluster import KMeans
from sqlalchemy.orm import Session
from .database import SessionLocal, engine

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File

from . import models, schemas, preprocessing

#this notebook will mostly be used for ML related tasks

# This is not the best practice, it is better to define the model via h5 file right away
# However there were some bugs that I haven't fixed yet, so I need to initialize the Model before loading the weights
# and on the other hand, This model initialization could also be useful for retraining after user add another new data
# Or the better approach is by using the AI service from either AWS or GCP making the prediction as an API 
# rather than putting the model here
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db = next(get_db())
df = preprocessing.preprocessing_data(db)
user_menu_quantities = df.groupby(["user_id", "menu_name"])["quantity"].sum().reset_index() #user total purchase per item
users = tf.data.Dataset.from_tensor_slices(dict(user_menu_quantities[["user_id", "menu_name", "quantity"]]))
menus = tf.data.Dataset.from_tensor_slices(dict(df[['menu_name']])) 
# Please do note that the menus here do not list the whole menus since it does seem that there is a bug that make me not being able to get all the menu set, possibly because of short await time

users = users.map(lambda x: {
    "user_id": x["user_id"],
    "menu_name": x["menu_name"],
    "quantity": float(x["quantity"])
})
menus = menus.map(lambda x: x["menu_name"])
shuffled = users.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = users.take(14_000) #train data used to train model
test = users.skip(14_000).take(3_937) #test data used to evaluate model

#batching
menu_names = menus.batch(1_000)
user_ids = users.batch(1_000).map(lambda x: x["user_id"])

#unique values
unique_menu_names = np.unique(np.concatenate(list(menu_names)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

#Recommendation Model
# Recommendation Model Intialiation
class MenuModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights. This weight tuning could be
    # utilized after doing the customer segmentation by adding more models and embeddings

    super().__init__()

    embedding_dimension = 64

    # User and movie models.
    self.menu_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_menu_names, mask_token=None),
      tf.keras.layers.Embedding(2031, embedding_dimension)
    ]) #used as an encoding method for each menu_ids
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(1001, embedding_dimension)
    ]) #used as an encoding method for each user_ids

    # A small model to take in user and menu embeddings and predict quantities-rankings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=menus.batch(128).map(self.menu_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the menu features and pass them into the menu model.
    menu_embeddings = self.menu_model(features["menu_name"])

    return (
        user_embeddings,
        menu_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and menu embeddings.
        self.rating_model(
            tf.concat([user_embeddings, menu_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("quantity")

    user_embeddings, menu_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, menu_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)
            
model = MenuModel(rating_weight=1.0, retrieval_weight=1.0)
# Build the model by calling it with example inputs
dummy_features = {
    "user_id": tf.constant(["dummy_user"]),
    "menu_name": tf.constant(["dummy_menu"])
}
model(dummy_features)
model.load_weights("model_weights.h5")

def load_model_weights():
  model.load_weights("model_weights.h5")

def predict_menu(user, top_n=3):
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # Recommends menus out of the entire menus dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((menus.batch(100), menus.batch(100).map(model.menu_model)))
    )

    # Get recommendations.
    _, titles = index(tf.constant([str(user)]))
    recommended_menu_ids = titles[0, :top_n * 100].numpy()  # Fetch more than top_n

    unique_menu_ids = set()  # To track unique menu recommendations
    unique_recommended_menu_ids = []

    for menuid in recommended_menu_ids:
        menuid_str = menuid.decode("utf-8")
        if menuid_str not in unique_menu_ids:
            unique_menu_ids.add(menuid_str)
            unique_recommended_menu_ids.append(menuid_str)

            if len(unique_recommended_menu_ids) >= top_n:
                break

    top_menu_recommendations = [
        {"rank": i + 1, "menu_id": menuid_str}
        for i, menuid_str in enumerate(unique_recommended_menu_ids)
    ]
    response = {
        "user_id": user,
        "top_menu_recommendations": top_menu_recommendations
    }
    return response

# Segmentation Model

# Customer Buying Behaviour frequent occasional consistent celining increasing
def logistic_weight(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def combine_values(values, L, k, x0):
    normalized_values = [v / 4.0 for v in values]  # Normalize values to 0-1 range
    weights = [logistic_weight(v, L, k, x0) for v in normalized_values]
    weighted_sum = sum(w * v for w, v in zip(weights, values))
    total_weight = sum(weights)
    combined_value = weighted_sum / total_weight
    return combined_value

# I realize that this is not the best practice, to train the model while also making it as an API, but since the wait time
# is not that long it is still acceptable, but the best practice is to have different functions for training and predict
def user_first_behaviour(user_id: str, n: int): # Frequent vs Occasional
  #user_id
  #n is number to choose n latest customer with id=user_id segmentation score you want to combine
  user_features = df.groupby(['user_id', 'month'])['quantity'].sum().reset_index()
  data_reshaped = user_features["quantity"].values.reshape(-1, 1)
  num_clusters = 5
  # Apply K-Means clustering
  kmeans = KMeans(n_clusters=num_clusters, random_state=42)
  clusters = kmeans.fit_predict(data_reshaped)
  cluster_mapping = {
    0: 0,
    3: 1,
    1: 2,
    4: 3,
    2: 4
  }
  user_features['cluster'] = clusters
  # Apply the cluster mapping to the 'cluster' column
  user_features['cluster'] = user_features['cluster'].map(cluster_mapping)
  user_rankings = user_features[user_features['user_id'] == user_id].sort_values(by='month', ascending=False)
  if len(user_rankings['cluster'].values) == 0:
    response = {"code": 404, "error": "could not find transaction", "message": "could not find transaction"}
  elif len(user_rankings['cluster'].values) < n:
    response = {"code": 404, "error": "n exceed user data length", "message": "n exceed user data length"}
  else:
    response = combine_values(user_rankings['cluster'].values[:n], 1.0, 1.0, 0.5)
  return response

def user_second_behaviour(user_id: str, n: int): 
  #we will be using SQLAlchemy if I have enough time to make the API
  user_features = df.groupby(['user_id', 'month'])['quantity'].sum().reset_index()
  all_user_ids = user_features['user_id'].unique()
  all_months = df['month'].unique()
  all_combinations = [(user_id, month) for user_id in all_user_ids for month in all_months]
  new_data = []
  for user_id, month in all_combinations:
    matching_rows = user_features[(user_features['user_id'] == user_id) & (user_features['month'] == month)]
    if matching_rows.empty:
        new_data.append({'user_id': user_id, 'month': month, 'quantity': 0.0})
    else:
        new_data.extend(matching_rows.to_dict('records'))
  new_user_features = pd.DataFrame(new_data)
  new_user_features['quantity_difference'] = new_user_features.groupby('user_id')['quantity'].diff()
  difference = new_user_features[new_user_features['month'] != 1]
  difference = difference.drop(columns=['quantity'])
  qty_reshaped = difference["quantity_difference"].values.reshape(-1, 1)
  num_clusters = 5
  kmeanscdi = KMeans(n_clusters=num_clusters, random_state=42)
  clusterscdi = kmeanscdi.fit_predict(qty_reshaped)

  # Add cluster labels to the DataFrame
  difference['cluster'] = clusterscdi

  # Map cluster labels to numerical values
  cluster_mapping = {
      2: 0,
      3: 1,
      0: 2,
      1: 3,
      4: 4
    }
  # Apply the cluster mapping to the 'cluster' column
  difference['cluster'] = difference['cluster'].map(cluster_mapping)
  user_rankings = difference[difference['user_id'] == user_id].sort_values(by='month', ascending=False)
  if len(user_rankings['cluster'].values) == 0:
    response = {"code": 404, "error": "could not find transaction", "message": "could not find transaction"}
  elif len(user_rankings['cluster'].values) < n:
    response = {"code": 404, "error": "n exceed user data length", "message": "n exceed user data length"}
  else:
    response = combine_values(user_rankings['cluster'].values[:n], 1.0, 1.0, 0.5)
  return response

def combine_behaviors(user_id: str, n: int):
    # Define mappings for first function
    first_output = round(user_first_behaviour(user_id, n))
    second_output = round(user_second_behaviour(user_id, n-1))
    first_mapping = {
        0: 'occasional',
        1: 'mid',
        2: 'frequent',        #does not actually mean normal but rather mid-scale
        3: 'frequent',
        4: 'frequent'
    }
    # Define mappings for second function outputs
    second_mapping = {
        0: 'declining',
        1: 'declining',
        2: 'declining',
        3: 'consistent',
        4: 'increasing'
    }
    # Combine the outputs
    combined_output = f"{second_mapping[second_output]} {first_mapping[first_output]}"
    return combined_output
