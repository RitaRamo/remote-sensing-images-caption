import math
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# important to read the models just once
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')


def compute_perplexity(sentence):
    tokens = tokenizer.encode(sentence)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
    return math.exp(loss / len(tokens))


def filter_best_caption(options, k1=2, k2=2):
    sentences = [[[], 0]]
    for position, tokens in enumerate(options):
        print("\npos and tokens", pos, tokens)
        new_sentences = []
        for sp, sentence in enumerate(sentences):
            print("sp and sentence", sp, sentence)
            best = dict()
            for token in set(tokens):
                best[token] = compute_perplexity(' '.join(sentence[0] + [token]))
                print("this is best tken", best)
            list_best = [k for k, v in sorted(best.items(), key=lambda item: item[1])]
            print("this is list best", list_best)
            for i in range(1, min(k1, len(list_best))):
                new_sentences.append([sentences[sp][0] + [list_best[i]], best[list_best[i]]])
                print("new sentece", new_sentences)
            sentences[sp][0].append(list_best[0])
            print("sentece append", sentences)
            sentences[sp][1] = best[list_best[0]]
            print("sentece append", sentences)

        sentences.extend(new_sentences)
        print("end of senteces", sentences)

        if len(sentences) > k2:
            print("é maior")
            sentences = [i for pos, i in enumerate(sorted(sentences, key=lambda item: item[1])) if pos < k2]
            print("senteces", sentences)
    sentences = [i[0] for i in sorted(sentences, key=lambda item: item[1])]
    print("senteces end", sentences)

    return sentences[0]


# the caption is represented as a sequence of candidate possibilities
generated_caption = [['This'],
                     ['is', 'sentence'],
                     ['a', 'in', 'is'],
                     ['nice', 'English', 'written', 'very'],
                     ['english', 'written', 'in', 'well'],
                     ['sentence', 'is', 'English', 'written'],
                     ['.', 'sentence']]
print(filter_best_caption(generated_caption))
