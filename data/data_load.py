import os
import json
from bs4 import BeautifulSoup

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
folder_path = 'all_speeches'


def compile_speeches():

    # Iterate over each file in the folder
    count = 0
    speech_dict = {}
    for filename in os.listdir(folder_path):
        
        print('Number ', count, ' of ', len(os.listdir(folder_path)))

        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.json'):

            # If the file is a JSON file, load the data
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            speech = json_data['anforande']['anforandetext']

            ## Text contains a lot of left over HTML code, so remove it
            soup = BeautifulSoup(speech, 'html.parser')
            text = soup.get_text()
            
            ## Removing STYLEREF och \* MERGEFORMAT from text
            text = text.replace('STYLEREF ', '')
            text = text.replace('\* MERGEFORMAT ', '')
            
            ## Lowercasing all tokens
            text = text.lower()
            
            ## Tokenizing the text
            tokens = tokenizer.tokenize(text)
        
            ## Cutting the speech of at 512 token length
            if len(tokens) <= 512:
                parsed_speech = ' '.join(tokens)
            else:
                token_sublist = [tokens[index] for index in range(0,512)]
                parsed_speech = ' '.join(token_sublist)

            speech_dict[count] = {}
            speech_dict[count]['tal'] = parsed_speech
            speech_dict[count]['parti'] = json_data['anforande']['parti']
            count += 1
            
    
    json_string = json.dumps(speech_dict, ensure_ascii=False)
    with open("preprocessed_speeches.JSON", "w", encoding="utf-8") as f:
        f.write(json_string)

compile_speeches()