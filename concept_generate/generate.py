import argparse
import json
import os

import requests


STAGES = ['early', 'intermediate', 'advanced', 'metastatic']


def _build_prompt(cancer: str, modal: str, stage: str, n: int) -> str:
    if modal == 'path':
        return (
            f'What are the morphological features in WSIs of {cancer} at the '
            f'{stage} stage of progression, please describe using keywords or '
            f'short sentences. Give {n} features and anwser the question with '
            f'the following format: 1.; 2.; 3.; ... .'
        )
    return (
        f'What are the radiological features in CT images of {cancer} at the '
        f'{stage} stage of progression, please describe using keywords or '
        f'short sentences. Give {n} features and anwser the question with the '
        f'following format: 1.; 2.; 3.; ... .'
    )


def _query_openai(prompt: str, model: str = 'gpt-4o') -> str:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set in environment.')
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        data=json.dumps({
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
        }),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def _query_gemini(prompt: str, model: str = 'gemini-2.0-flash') -> str:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY not set in environment.')
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{model}:generateContent?key={api_key}'
    )
    response = requests.post(
        url,
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'contents': [{'parts': [{'text': prompt}]}]}),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['candidates'][0]['content']['parts'][0]['text']


def clean_ans(text: str, N: int) -> list:
    """Original split-based parser.  Returns exactly N items.

    Kept verbatim from the research code so candidate pools match bit-for-bit
    on identical LLM outputs.
    """
    concept_list = []
    for i in range(N):
        concept = text.split(f'{i + 1}.')[1].split(f'{i + 2}.')[0].strip()
        if i == N - 1:
            if '\n' in concept:
                concept = concept.split('\n')[0]
        concept = concept.replace('*', '')
        concept_list.append(concept)
    return concept_list


def save_dictionary_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', required=True, help='e.g. "gastric tumor"')
    parser.add_argument('--modal', required=True, choices=['path', 'rad'])
    parser.add_argument('--n', type=int, default=250,
                        help='Number of concepts requested per stage.')
    parser.add_argument('--backend', choices=['openai', 'gemini'], default='openai')
    parser.add_argument('--out_dir', default='.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.cancer}_{args.modal}.json')

    model_res = {}
    for stage in STAGES:
        prompt = _build_prompt(args.cancer, args.modal, stage, args.n)
        text = _query_openai(prompt) if args.backend == 'openai' else _query_gemini(prompt)
        concept_list = clean_ans(text, args.n)
        model_res[stage] = concept_list
        save_dictionary_json(model_res, out_path)
        print(f'[{stage}] {len(concept_list)} concepts -> {out_path}')


if __name__ == '__main__':
    main()
