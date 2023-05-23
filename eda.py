# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
import datasets

random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', '']

augmentation_targets = ["sentence", "premise", "hypothesis", "sentence1", "sentence2", "question", "sentence", "question1", "question2"]

def eda_one_batch(examples, num_aug=9):
    new_examples = {}
    shortest_augs = 0
    eda_results = {}

    lengths = 0
    shortest_augmentation_lengths = []

    for specific_feature in examples:
        lengths = len(examples[specific_feature])
        if specific_feature not in new_examples:
            new_examples[specific_feature] = []
        if specific_feature not in augmentation_targets:
            eda_results[specific_feature] = examples[specific_feature]
        else:
            eda_results[specific_feature] = []
            for sent_index, sentence in enumerate(examples[specific_feature]):
                eda_sentences = eda(sentence, num_aug=num_aug)
                eda_results[specific_feature].append(eda_sentences)
                if len(shortest_augmentation_lengths) <= sent_index:
                    shortest_augmentation_lengths.append(len(eda_sentences))
                else:
                    new_length = len(eda_sentences)
                    if new_length < shortest_augmentation_lengths[sent_index]:
                        shortest_augmentation_lengths[sent_index] = new_length

    for index in range(len(shortest_augmentation_lengths)):
        for feature in eda_results:
            if feature not in augmentation_targets:
                for i in range(shortest_augmentation_lengths[index]):
                    new_examples[feature].append(eda_results[feature][index])
            else:
                for i in range(shortest_augmentation_lengths[index]):
                    new_examples[feature].append(eda_results[feature][index][i])
        # print(new_item)
    return new_examples

def eda_dataset(dataset: datasets.Dataset, num_augs=9):
    edaed_dataset = dataset.map(eda_one_batch, batched=True, fn_kwargs={'num_aug': num_augs})
    return edaed_dataset


def eda_dataset_naive(dataset: datasets.Dataset, num_augs=9):

    eda_features = []
    num_items = dataset.num_rows
    all_features = list(dataset.features.keys())

    for feature in dataset.features:
        if isinstance(dataset.features[feature], datasets.Value) and dataset.features[feature].dtype == 'string':
            eda_features.append(feature)
    # print(eda_features)

    eda_new_items = []
    for item in dataset:
        eda_results = {}
        no_aug = False
        for specific_feature in item:
            if specific_feature in eda_features:
                sentence = item[specific_feature]
                eda_sentences = eda(sentence, num_aug=num_augs)
                if len(eda_sentences) <= 0:
                    no_aug = True
                # print(sentence, specific_feature, len(eda_sentences))

                eda_results[specific_feature] = eda_sentences
            else:
                eda_results[specific_feature] = item[specific_feature]

        if no_aug:
            continue

        for index in range(num_augs):
            new_item = {}
            for feature in eda_results:
                if feature not in eda_features:
                    if feature != 'id':
                        new_item[feature] = eda_results[feature]
                    else:
                        new_item[feature] = num_items
                        num_items += 1
                else:
                    new_item[feature] = eda_results[feature][index]
            # print(new_item)
            eda_new_items.append(new_item)

    for new_item in eda_new_items:
        dataset = dataset.add_item(new_item)

    return dataset

#cleaning up text
import re
def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()
    if not line:
        return clean_line

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)
    if num_words <= 0:
        return [sentence]

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1

    #sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    #ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    #rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    #rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    #append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences