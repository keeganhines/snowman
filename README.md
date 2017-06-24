# Snowman

This project focuses on the use of convolutional neural networks for the identification of malicious URLs and the presence of Domain Generation Algorithms. The modeling work is largely inspired by
  * http://www.aclweb.org/anthology/D14-1181
  * https://arxiv.org/abs/1702.08568
  * https://arxiv.org/abs/1610.01969 

Further, the resulting models are deployed through a REST API so that model scoring and prediction is exposed for generic use.

# Contents
1. [Use](#use)
2. [Installation](#installation)
3. [Deployment](#deployment)
4. [Backend](#backend)
5. [Modeling](#modeling)

## Use

This model is deployed through Heroku and can be accessed through a REST API. By hitting the endpoint `/model`, you are able to supply a query domain that you suspect may be malware related, and will receive a score from the model.

```
curl http://afternoon-bastion-75939.herokuapp.com/model -d "url=zis32msdi3.com" -X PUT
```
This request returns

```
{"input query": "zis32msdi3.com", "score": "0.19122"}
```
Because I'm using the Heroku free tier, the application will go into idle sleep after 30 minutes without any requests. If application is not responsive, it is probably asleep. But good news is that your request should wake it back up and it'll be back in action within a few seconds.

## Installation

Set up and activate a virtual environment.

```
cd some-file-path/snowman
virtualenv env
source env/bin/activate
```

Then use pip to install required packages.

```
pip install -r requirements.txt
```

## Deployment

My deployment strategy uses heroku. You can quickly get a free account which access to their free-tier deployments. And you will download the heroku CLI which allows you to deploy apps to heroku and interact with your deployments through your command line (no ssh and scp of materials).

```
cd some-file-path/snowman
heroku create
```
This will kick off various set up scripts. Then then deploy code to herko by pushing to a git branch called heroku (that get's made for you). 

```
git push heroku master
```

This pushes out your materials to your heroku instance (called a dyno) and will install all required packages on your dyno. It will also startup your app according to the specifications set in the Procfile. You can check on the progress of set up using

```
heroku logs
```

Once complete, your app will be serving on some randomly-generated subdomain such as "afternoon-bastion-75939.herokuapp.com". You can then query whatever endpoints and services you have set up for yourself. One note about the herkou free tier - because it's free they will auto-hibernate your dyno after 30 minutes of no requests to your app. Once in hibernation, your app won't be able to quickly respond to new requests. But, any new request that comes in while it is asleep will have the affect of waking up your dyno. So then the app gets kicked back into action and will be able to handle requests again in a few seconds.

## Backend
For deploying the model behind a backend REST API, I'm using python's Flask- more specifically, I'm mostly using flask_restful for defining endpoints and actions. A previously-trained model is deserialized and loaded. Then when the endpoint "/model" is queried with a PUT or a POST, the provided query string is run against the model for scoring. As described above, hosting is done with heroku.

## Modeling

The model I'm using here is a convolutional neural network adapted for text classification. The convolutional kernels, pooling layers, and activations are all fairly standard. I suppose the slightly uncommon thing is how to use convolution with text data - this is why the first layer in the network is an embedding layer. Each character in a string is represented as a point in a low dimensional vector space. Thus, a whole string (character sequence) gives us a numeric array - to which I zero-pad to standard the sizes across all strings in the training data. Then, to this numeric array, we can apply all the convolutions we desire. More details about these ideas can be found in the Yoon Kim paper (above), where this kind of network was applied to movie review sentitment classification.