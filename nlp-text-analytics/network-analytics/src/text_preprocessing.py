#############################
# Author: S. A. Owerre
# Date modified: 26/08/2022
# Class: Text Preprocessing
#############################

# Filter warnings
import warnings

warnings.filterwarnings('ignore')
import string
import ast


class TextPreprocessing:
    """A class for preprocessing text data."""

    def __init__(self):
        """Define parameters."""
        pass

    def string_to_list(self, x):
        """Convert string representation of list into a list of strings.

        Example:
        input: x = '["a","b","c"]'
        output: x = ["a","b","c"]

        Parameter
        ---------
        x: string representation of list.

        Returns
        -------
        list of strings.
        """
        return ast.literal_eval(x)

    def tokenizer(self, text):
        """Tokenizer removes special characters and punctuations.

        Parameters
        ----------
        text: a string of texts or sentences

        Returns
        -------
        text without special characters and punctuations
        """
        symbols = string.punctuation + '0123456789\n'
        nospe_char = [char for char in text if char not in symbols]
        return ''.join(nospe_char)

    def split_extract(self, text, split_on=None):
        """Split text and extract the first element.

        Parameters
        ----------
        text: string of texts or sentences
        split_on: string to split on

        Returns
        -------
        first element in text
        """
        text = text.split(split_on)
        return text[0]
