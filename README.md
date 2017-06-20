# Snowman

This project focuses on the use of convolutional neural networks for the identification of malicious URLs and the presence of Domain Generation Algorithms. The modeling work is largely inspired by
  * http://www.aclweb.org/anthology/D14-1181
  * https://arxiv.org/abs/1702.08568
  * https://arxiv.org/abs/1610.01969 

Further, the resulting models are deployed through a REST API so that model scoring and prediction is exposed for generic use.

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

## Deployment