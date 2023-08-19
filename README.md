# Food_Audric
 Technical test that involves building machine learning models for a recommendation system.

Please install all the dependency first using
pip install -r requirements.txt

Most of the explanations were made on notebooks. It has more models and some other things that is not adapted yet to the API

To access the API, clone this then add a converted (to sqlite db) data.csv you have given, name it as food_app.db put it in food_app folder
After that cd to Food_Audric and on cmd type
uvicorn food_app.main:app --reload
to start the server

You could either use postman or access http://127.0.0.1:8000/docs/ to test the APIs. I have not made any automated test for this project.

You could see this project documentation on https://docs.google.com/document/d/1N4oF_i4r8j3YpXRAyz5vNyQ3Xt91i37EpnYNlJ7YyTo/edit?usp=sharing