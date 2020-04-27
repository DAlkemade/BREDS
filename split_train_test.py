from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from breds.config import read_objects_of_interest


def main():
    parser = ArgumentParser()
    parser.add_argument('--objects', required=True, type=str)
    args = parser.parse_args()
    objects_path = args.objects
    names = list(read_objects_of_interest(objects_path))
    train, test = train_test_split(names, test_size=.1)
    save(train, objects_path, 'train')
    save(test, objects_path, 'test')


def save(objects: list, fname_original: str, tag: str):
    f_components = fname_original.split(".")
    assert len(f_components) == 2
    fname = f'{f_components[0]}_{tag}.{f_components[1]}'
    with open(fname, 'w') as f:
        for object in objects:
            f.write(f'{object}\n')


if __name__ == "__main__":
    main()
