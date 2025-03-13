import requests

# url = "https://api.spectator.earth/imagery/?bbox=28.02,-26.21,28.07,-26.18&api_key=7WRpYhUWE6KYE9EAENvwTt&date_from=2025-02-01&date_to=2025-03-06"

# payload = {}
# headers = {}

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)

# url = (
#     "https://api.spectator.earth/imagery/39740731/files/?api_key=7WRpYhUWE6KYE9EAENvwTt"
# )

# payload = {}
# headers = {}

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)

import ee

ee.Authenticate()
# ee.Initialize(project="my-project")
