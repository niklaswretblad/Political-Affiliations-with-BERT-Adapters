import os
import json
from bs4 import BeautifulSoup

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
folder_path = 'data/all_speeches'

# TODO - Ta bort TALMANNEN tal? 


def compile_speeches():

    # Iterate over each file in the folder
    speech_dict = {}
    print(os.getcwd())
    for i, filename in enumerate(os.listdir(folder_path)):

        if i % 500 == 0:
            print('Number ', i, ' of ', len(os.listdir(folder_path)))

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
            #tokens = tokenizer.tokenize(text)
            tokens = text.split(' ')
        
            ## Cutting the speech of at 512 token length
            if len(tokens) <= 512:
                parsed_speech = ' '.join(tokens)
            else:
                token_sublist =  tokens[0: 512]
                parsed_speech = ' '.join(token_sublist)

            speech_dict[i] = {}
            speech_dict[i]['text'] = parsed_speech
            speech_dict[i]['label'] = json_data['anforande']['parti']
            
    
    json_string = json.dumps(speech_dict, ensure_ascii=False)
    with open("preprocessed_speeches.json", "w", encoding="utf-8") as f:
        f.write(json_string)

def main():
    compile_speeches()
    
if __name__ == '__main__':
    main()

