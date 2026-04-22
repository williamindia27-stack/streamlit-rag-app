import requests
response = requests.get("https://api.github.com")

print(response.status_code)
print(response.text)

#What’s happening
#get() → asks the API for data
#status_code → tells if it worked
#text → raw response

#Json
import requests

response = requests.get("https://api.github.com")

data = response.json()

print(data)

#example crypto
import requests

url = "https://api.coindesk.com/v1/bpi/currentprice.json"

response = requests.get(url)
data = response.json()

price = data["bpi"]["USD"]["rate"]

print("Bitcoin price:", price)

#Send data
import requests

url = "https://httpbin.org/post"

data = {"name": "William"}

response = requests.post(url, json=data)

print(response.json())

#Headers require authentication
headers = {
    "Authorization": "Bearer YOUR_API_KEY"
}

response = requests.get("https://api.example.com", headers=headers)

#Handle errors
if response.status_code == 200:
    print("Success")
else:
    print("Error:", response.status_code)

#Real agent use case
def get_crypto_price():
    import requests
    url = "https://api.coindesk.com/v1/bpi/currentprice.json"
    data = requests.get(url).json()
    return data["bpi"]["USD"]["rate"]

print(get_crypto_price())