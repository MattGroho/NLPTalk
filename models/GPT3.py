import openai


class GPT3:

    doc_data, type = None, None

    def __init__(self, model_type, doc_data):
        self.type = model_type
        self.doc_data = doc_data

    def evaluate(self, text):
        openai.api_key = ''

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
                model='curie',
                question=text,
                documents=self.doc_data,
                examples_context="When was etown founded?",
                examples=[["Elizabethtown College was founded in 1899.", "Etown was founded in 1899."]],
                max_tokens=30,
                stop=['\n', '<|endoftext|>']
            )
            return response['answers'][0]
