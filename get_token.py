import argparse
import requests
import json
from base64 import b64encode 


base_url = "https://api.joinparrot.ai/v1"

def get_token():
    """
    This function performs addition of two numbers taken as command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Get Token for Parrot Host")
    parser.add_argument("-username", type=str, help="Your Parrot username")
    parser.add_argument("-password", type=str, help="Your Parrot password")

    args = parser.parse_args()

    url = f"{base_url}/user/login"
    # Authorization token: we need to base 64 encode it 
    # and then decode it to acsii as python 3 stores it as a byte string
    def basic_auth(username, password):
        token = b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
        return f'Basic {token}'

    #then connect
    headers = { 'Authorization' : basic_auth(args.username, args.password) }        
    response = requests.post(url, headers= headers)

    # obtain the token
    token = response.json()["data"]["access_token"]
    print(token)
    return token

if __name__ == "__main__":
    get_token()