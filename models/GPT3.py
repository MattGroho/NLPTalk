import openai


class GPT3:

    doc_data, type = None, None

    def __init__(self, model_type, doc_data):
        self.type = model_type
        self.doc_data = doc_data

    def evaluate(self, text):
        openai.api_key = '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'

        # Initialize model based on passed parameters
        if self.type == 'GPT3':
            response = openai.Completion.create(
                engine="davinci-instruct-beta",
                prompt=text,
                temperature=0.5,
                max_tokens=30,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].text.split('\n').split('.')[0]
        elif self.type == 'ETOWN':
            response = openai.Answer.create(
                search_model="davinci",
                file=open('data/EtownDocData.txt', 'r'),
                #documents=self.doc_data,
                model='curie',
                question=text,
                examples_context="Elizabethtown College was founded in 1899.",
                examples=[["When was Elizabethtown College founded?", "When was etown founded?"]],
                max_tokens=30,
                stop=["\n", "<|endoftext|>"]
            )
            return response
