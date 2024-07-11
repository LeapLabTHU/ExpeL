import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = ''

def test_openai_api():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello to the world"}
            ]
        )
        print("API call successful. Response:")
        print(response.choices[0].message['content'].strip())
    except Exception as e:
        print(f"API call failed: {e}")

if __name__ == "__main__":
    test_openai_api()
