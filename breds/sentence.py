import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from size_comparisons.scraping.lengths_regex import LengthsFinderRegex

from breds.config import Config

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

# tokens between entities which do not represent relationships
bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
# stopwords = stopwords.words('english')
# TODO check wehther this was a good idea. but e.g. 'is' is in there, which is bad
stopwords = []
not_valid = bad_tokens + stopwords


def tokenize_entity(entity):
    parts = word_tokenize(entity)
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    locations = []
    e_parts = tokenize_entity(entity_string)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return e_parts, locations


class EntitySimple:
    def __init__(self, _e_string, _e_parts, _e_type, _locations):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations

    def __hash__(self):
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other):
        return self.string == other.string and self.type == other.type


class EntityLinked:
    def __init__(self, _e_string, _e_parts, _e_type, _locations, _url=None):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations
        self.url = _url

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return self.url == other.url


class Relationship:
    def __init__(self, _sentence, _before, _between, _after, _ent1, _ent2,
                 e1_type, e2_type):
        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.e1 = _ent1
        assert type(_ent2) is float
        self.e2 = _ent2
        self.e1_type = e1_type
        self.e2_type = e2_type

    def __eq__(self, other):
        if self.e1 == other.e1 and self.before == other.before and \
                        self.between == other.between \
                and self.after == other.after:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2) ^ hash(self.before) ^ \
               hash(self.between) ^ hash(self.after)


class Sentence:

    def __init__(self, sentence_no_tags: str, e1_type, e2_type, max_tokens, min_tokens,
                 window_size, goal_object: str, pos_tagger=None, config: Config = None):

        if config.e1_type != 'OBJECT' or config.e2_type != 'NUMBER':
            raise RuntimeError("This function is only suitable for object-numer combinations")
        self.relationships = list()
        self.tagged_text = None

        # determine which type of regex to use according to
        # how named-entities are tagged
        numbers_regex = re.compile(rf'[0-9]+\.?[0-9]*')
        objects_regex = re.compile(rf'({goal_object})s?')

        # find named-entities
        # numbers = []
        # for m in re.finditer(numbers_regex, sentence_no_tags):
        #     numbers.append(m)
        # if goal_object == 'tiger':
        #     print(sentence_no_tags)
        finder = LengthsFinderRegex(sentence_no_tags)
        numbers, _ = finder.find_all_matches()
        # if goal_object == 'statue of liberty':
        #     # print('sentence:' + sentence_no_tags)
        #     if 'measures 305' in sentence_no_tags:
        #         print(repr(sentence_no_tags))
        #         print(f'numbers: {numbers}')

        objects = []
        for m in re.finditer(objects_regex, sentence_no_tags):
            objects.append(m)


        if len(numbers) >= 1 and len(objects) >= 1:
            text_tokens = word_tokenize(sentence_no_tags)

            # extract information about the entity, create an Entity instance
            # and store in a structure to hold information collected about
            # all the entities in the sentence
            entities_info = set()
            for x in range(0, len(numbers)):
                if config.tag_type == "simple":
                    # entity = numbers[x].group()
                    entity = numbers[x][0]
                    number_in_meters = numbers[x][1]
                    e_string = entity
                    e_type = 'NUMBER'
                    e_parts, locations = find_locations(e_string, text_tokens)
                    e = EntitySimple(number_in_meters, e_parts, e_type, locations)
                    entities_info.add(e)
            for x in range(0, len(objects)):
                if config.tag_type == "simple":
                    entity_clean = objects[x].group(1)
                    entity_raw = objects[x].group()
                    e_string = entity_raw
                    e_type = 'OBJECT'
                    e_parts, locations = find_locations(e_string, text_tokens)
                    e = EntitySimple(entity_clean, e_parts, e_type, locations)
                    entities_info.add(e)


            # create an hash table:
            # - key is the starting index in the tokenized sentence of an entity
            # - value the corresponding Entity instance
            locations = dict()
            for e in entities_info:
                for start in e.locations:
                    locations[start] = e

            # look for pair of entities such that:
            # the distance between the two entities is less than 'max_tokens'
            # and greater than 'min_tokens'
            # the arguments match the seeds semantic types
            sorted_keys = list(sorted(locations))
            for i in range(len(sorted_keys)-1):
                distance = sorted_keys[i+1] - sorted_keys[i]
                e1 = locations[sorted_keys[i]]
                e2 = locations[sorted_keys[i+1]]

                if max_tokens >= distance >= min_tokens and e1.type == e1_type \
                        and e2.type == e2_type:

                    # ignore relationships between the same entity
                    if config.tag_type == "simple":
                        if e1.string == e2.string:
                            continue
                    elif config.tag_type == "linked":
                        if e1.url == e2.url:
                            continue

                    # run PoS-tagger over the sentence only once
                    if self.tagged_text is None:
                        # split text into tokens and tag them using NLTK's
                        # default English tagger
                        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/
                        # english.pickle'
                        self.tagged_text = pos_tagger.tag(text_tokens)

                    before = self.tagged_text[:sorted_keys[i]]
                    before = before[-window_size:]
                    between = self.tagged_text[sorted_keys[i] +
                                               len(e1.parts):sorted_keys[i+1]]
                    after = self.tagged_text[sorted_keys[i+1]+len(e2.parts):]
                    after = after[:window_size]

                    # ignore relationships where BET context is only stopwords
                    # or other invalid words
                    if all(x in not_valid for x in
                           text_tokens[sorted_keys[i] + len(e1.parts):sorted_keys[i + 1]]):
                        continue

                    if config.tag_type == "simple":
                        r = Relationship(
                            sentence_no_tags, before, between, after, e1.string,
                            e2.string, e1_type, e2.type
                        )

                    elif config.tag_type == "linked":
                        r = Relationship(
                            sentence_no_tags, before, between, after, e1.url, e2.url,
                            e1.type, e2.type
                        )

                    self.relationships.append(r)
