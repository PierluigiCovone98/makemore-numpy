"""Character-Level language model that process and generate text one character at time (autoregressive).

Text files are assumed to be "one word for line".
"""
from pathlib import Path

# The "TXT" file become a list of strings.
type Stoi =  dict[str, int]
type Itos =  dict[int, str]

# Constants
BOUNDARY_TOKEN: str = '.'

def read_dataset( dataset_path: str ) -> list[str]:
    """Read the dataset transforming it in a list of strings."""

    # "Path" is more useful than "OS" (for me)
    path = Path(dataset_path)

    return path.read_text().splitlines()




def build_vocab( dataset: list[str] ) -> tuple[Stoi, Itos]:
    """Takes in input the read dataset (list of strings) and returns
    dictionaries 'stoi' and 'itos', that are required to encode a character to it's integer
    and an integer to the corresponding charachter, respectively.
    
    In addition to the alphabet retrieved from the dataset, a speciale token '.' is inserted
    to mark the start and the end of a word; the corresponding integer to it is: 0.

    The first element that is returned is "stoi";
    the second one is "itos".
    """

    sorted_alphabet = sorted(set(''.join(dataset)))
   
    stoi : Stoi = {}
    itos : Itos = {}

    # I do not expect "len_alphabet" to be high.
    for i, s in enumerate(sorted_alphabet, start=1):
        stoi[s] = i
        itos[i] = s
    
    # By convention, it is put at the end of the dictionary.
    stoi[BOUNDARY_TOKEN] = 0
    itos[0] = BOUNDARY_TOKEN

    return (stoi, itos)


    # Next operations:
    #   encode(stoi: Stoi, ...)
    #   decode(itos: Itos, ...)
    



def main():
    print(build_vocab( read_dataset("data/names.txt") ) )

if __name__ == "__main__":
    main()