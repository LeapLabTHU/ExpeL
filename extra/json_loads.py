import json

json_string = '[{"name": "John", "age": 30, "city": "New York"}, {"name": "Carter", "age": 23, "city": "Ankara"}]'

try:
    data = json.loads(json_string)
    print(data)
except json.JSONDecodeError:
    print("Failed to decode JSON")
