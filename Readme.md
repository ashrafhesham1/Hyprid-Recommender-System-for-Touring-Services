# Hyprid Recommender System for Touring Services

## Table of Contents

- [Description](#description)
- [Architecture](#archticture)
  - [General Architecture](#general-archticture)
  - [Making Predictions for Unseen Users](#making-predictions-for-unseen-users)
- [How to Use It](#how-to-use-it)
- [Technologies](#technologies)

## Description

A weighted hyprid recommender system for touring servicrs that combines content-based recommendations with collaborative filltering recommendations using learned weights with the ability to make predictions for unseen users on-the-fly given their ratings.

## Archticture

### General Archticture

![Archticture diagram](./reports/Recommender%20Archticture%201.png)

As Seen in the figure above, the system consists of two models:

1- **KNN-Based Content-Based** model which estimates the rating from a given user to a given item by measuring the **content-based similarity** between this item and all the items the user has rated before and calculates a **similarity weighted average** among the top-k nearest neighbor.

2- **Matrix-Factorization-Based** model that uses **singular value decomposition SVD** and **stochastic gradient descent SGD** to represent users and items as latent vectors in a common space which are used later to estimate the required ratings.

The estimations given by the two models are combined later by taking a **weighted average** of them in which the weights are learned using a **linear regression model** that has trained to map the predictions of the two models into a final prediction.

### Making predictions for unseen users

The method described earlier is only for making predictions for users that have been in the trainset, However, it doesn't work for new users even though the Content-based model can make predictions for a new user as it requires the ratings of the user to be given with each call but the SVD-Based model can't natively handle new users.

`Note: SVD can fit new users by adding a new row to the matrix and running SGD in order to find the representation of the new user but it tends not to perform well which was the case with this system.`

The figure below shows how the model got adjusted to be able to make predictions for new users.

![Archticture diagram](./reports/Recommender%20Archticture%202.png)

as the figure shows the adjusted version uses the SVD model to compute a full ratings matrix for all users and items and then uses that as baises for a KNN-Based model which compute the estimate of the rating as a similarity-weighted average among the top-K nearest neighbors then this predicted rating is passed to a baseline model which adjusts the rating to reflect the difference in bias between the user and the neighbors and compute the final estimation of the collaborative filtering component of the model.

## Evaluation

### Hit rate score
![Archticture diagram](./reports/HIT%20RATE.png)

## How to use it

1- In the `.env` file setup **items_path** and **ratings_path**.

`Note: the items must have the shape descriped in the content-based class and the ratings on the form (user, item, rating).`

2- Instantiate the **model** class and call the method **fit** to fit the model to the data.

3- Use the predict method to predict ratings from users to items.

`Note: predict method takes user_id and item_id as arguments in case of predicting a rating for an existing user and takes a dictionary of ratings that maps items to ratings when working with unseen users`.

## Technologies

scikit-surprise

scikit-learn
