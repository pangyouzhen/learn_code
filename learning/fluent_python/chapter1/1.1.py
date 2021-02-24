import collections
from typing import List

Card = collections.namedtuple("Card", ['rank', 'suit'])


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards: List[Card] = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, item):
        return self._cards[item]


suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)


def spades_high(card: Card) -> int:
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]


deck = FrenchDeck()
print(deck)

print(deck[0])
print(deck[2])
# in 也会调用 getitem方法
print(Card("Q", "hearts") in deck)

from random import choice

print(choice(deck))

# for card in sorted(deck, key=spades_high):
#     print(card)
