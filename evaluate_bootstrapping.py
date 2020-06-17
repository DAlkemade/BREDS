import fileinput
from argparse import ArgumentParser
from typing import Dict
import re
import numpy as np
from breds.config import parse_objects_from_seed
from matplotlib import pyplot as plt
import pprint

pp = pprint.PrettyPrinter()


def main():
    """Evaluate the output of a bootstrapping run."""
    parser = ArgumentParser()
    parser.add_argument('--relationships', required=True, type=str)

    seeds: set = parse_objects_from_seed('data_numeric/seeds_positive.txt').union(
        parse_objects_from_seed(
            'data_numeric/seeds_negative.txt'))

    args = parser.parse_args()
    non_decimal = re.compile(r'[^\d.]+')
    tuples = set()
    for line in fileinput.input(args.relationships, openhook=fileinput.hook_encoded("utf-8")):
        if line.startswith("instance"):
            entity = line.split('\t')[0][10:]
            size = float(line.split('\t')[1])
            confidence_str = line.split('\t')[2]
            confidence = float(non_decimal.sub('', confidence_str))
            tuples.add((entity, size, confidence))

    # Remove objects that were in OG seed
    tuples = [tuple for tuple in tuples if tuple[0] not in seeds]
    tuples_high_conf = [tuple for tuple in tuples if tuple[2] > .4]

    # tuples_lookup = create_lookup(tuples)
    tuples_lookup_high_conf = create_lookup(tuples_high_conf)
    pp.pprint(tuples_lookup_high_conf)


def create_lookup(tuples) -> dict:
    tuples_lookup: Dict[str, set] = dict()
    for tuple in tuples:
        entity = tuple[0]
        size = tuple[1]
        confidence = tuple[2]
        new_tuple = (size, confidence)
        try:
            tuples_lookup[entity].add(new_tuple)
        except KeyError:
            tuples_lookup[entity] = {new_tuple}
    print(f'Number of entities with at least one size: {len(tuples_lookup.keys())}')

    averages = list()
    for e, v in tuples_lookup.items():
        s, confs = zip(*v)
        averages.append(np.mean(s))

    print(f'biggest object: {list(tuples_lookup.items())[np.argmax(averages)]}')
    bins = np.linspace(0,max(averages),50)
    plt.hist(averages, bins=bins)
    plt.xlabel('Mean object size')
    plt.title('All objects')
    plt.show()
    bins = np.linspace(0, 20, 50)
    plt.hist(averages, bins=bins)
    plt.title('Small objects zoom')
    plt.xlabel('Mean object size')
    plt.show()

    return tuples_lookup


if __name__ == "__main__":
    main()