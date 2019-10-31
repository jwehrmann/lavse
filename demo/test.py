import requests
import json

url = 'http://0.0.0.0:5000/api/'

data = {'query': 'a boy is walking down the street'}
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
d = json.loads(r.text)
print(d['image'])
