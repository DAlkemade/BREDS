import neuralcoref
import spacy


def print_clusters(text: str):
    print('\n')
    print(text)
    doc = nlp(
        text)

    for ent in doc.ents:
        print(f'{ent}: {ent._.coref_cluster}')
    print(doc._.coref_clusters)

    print(doc._.coref_resolved)

spacy.require_gpu()
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
print_clusters('My sister has a dog. She loves him.')

print_clusters('Angela lives in Boston. She is quite happy in that city.')

print_clusters(
    'The Bengal and Siberian tigers are amongst the tallest cats in shoulder height. They are also ranked among the biggest cats that have ever existed reaching weights of more than 350 kg (770 lb).')

print_clusters("The lion (Panthera leo) is a species in the family Felidae; it is a muscular, deep-chested cat with a short, rounded head, a reduced neck and round ears, and a hairy tuft at the end of its tail. Adult male lions have a prominent mane, which is the most recognisable feature of the species. With a typical head-to-body length of 184–208 cm (72–82 in) they are larger than females at 160–184 cm (63–72 in).")
