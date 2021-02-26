from torchtext.datasets.text_classification import *
from torchtext.datasets.text_classification import _csv_iterator, _create_data_from_iterator
from torchtext.datasets import text_classification
import copy
from torchtext.datasets.text_classification import _setup_datasets

_setup_datasets_copy = copy.deepcopy(_setup_datasets)


def my_setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None, include_unk=False, downloads=False, path=None):
    dataset_tar = None
    if downloads is True:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
    if downloads is False and path is None:
        assert "必须指定文件路径"
    elif downloads is False and path is not None:
        dataset_tar = path
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))


text_classification._setup_datasets = my_setup_datasets
text_classification.DATASETS = {
    'AG_NEWS': AG_NEWS,
    'SogouNews': SogouNews,
    'DBpedia': DBpedia,
    'YelpReviewPolarity': YelpReviewPolarity,
    'YelpReviewFull': YelpReviewFull,
    'YahooAnswers': YahooAnswers,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'AmazonReviewFull': AmazonReviewFull
}

text_classification.LABELS = {
    'AG_NEWS': {0: 'World',
                1: 'Sports',
                2: 'Business',
                3: 'Sci/Tech'},
    'SogouNews': {0: 'Sports',
                  1: 'Finance',
                  2: 'Entertainment',
                  3: 'Automobile',
                  4: 'Technology'},
    'DBpedia': {0: 'Company',
                1: 'EducationalInstitution',
                2: 'Artist',
                3: 'Athlete',
                4: 'OfficeHolder',
                5: 'MeanOfTransportation',
                6: 'Building',
                7: 'NaturalPlace',
                8: 'Village',
                9: 'Animal',
                10: 'Plant',
                11: 'Album',
                12: 'Film',
                13: 'WrittenWork'},
    'YelpReviewPolarity': {0: 'Negative polarity',
                           1: 'Positive polarity'},
    'YelpReviewFull': {0: 'score 1',
                       1: 'score 2',
                       2: 'score 3',
                       3: 'score 4',
                       4: 'score 5'},
    'YahooAnswers': {0: 'Society & Culture',
                     1: 'Science & Mathematics',
                     2: 'Health',
                     3: 'Education & Reference',
                     4: 'Computers & Internet',
                     5: 'Sports',
                     6: 'Business & Finance',
                     7: 'Entertainment & Music',
                     8: 'Family & Relationships',
                     9: 'Politics & Government'},
    'AmazonReviewPolarity': {0: 'Negative polarity',
                             1: 'Positive polarity'},
    'AmazonReviewFull': {0: 'score 1',
                         1: 'score 2',
                         2: 'score 3',
                         3: 'score 4',
                         4: 'score 5'}
}
