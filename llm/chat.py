from openai import OpenAI, AzureOpenAI
from argparse import ArgumentParser
import os
import dotenv

def main():
    dotenv.load_dotenv()

    parser = ArgumentParser()
    parser.add_argument("--api_version", type=str, default=None)
    parser.add_argument("--stream", action='store_true')
    parser.add_argument('--api_type', type=str, default="chat", choices=["chat", "responses"])
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if args.api_version is not None:
        client = AzureOpenAI(base_url=base_url, api_version=args.api_version, api_key=api_key)
    else:
        client = OpenAI(base_url=base_url, api_key=api_key)

    messages = [
        {"role": "user", 
        #  "content": "Hi! How are you today?"
        "content": "Count from 1 to 100 in a line separated by comma."
        },
    ]
    if args.api_type == "responses":
        response = client.responses.create(
            model=args.model_name,
            input=messages
        )
        print(response)
    else:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            stream=args.stream
        )
        if args.stream:
            for chunk in response:
                print(chunk.choices[0].delta.content, end="", flush=True)
            print()
        else:
            print(response.choices[0].message.content)


if __name__ == "__main__":
    main()