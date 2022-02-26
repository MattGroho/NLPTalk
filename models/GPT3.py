import openai


class GPT3:
    def __init__(self):
        pass

    def evaluate(self, text):
        openai.api_key = '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'
        response = openai.Completion.create(
            engine="davinci-instruct-beta",
            prompt=text,
            temperature=0.5,
            max_tokens=30,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response.choices[0].text.split('.')
        return content[0]
