import nltk

text = "That U.S.A. poster-print costs $12.40..."

pattern = r"""(?x)
(                               # Outer capturing group for the full match
    (?:[A-Z]\.)+                # Abbreviations like U.S.A.
    | \w+(?:-\w+)* # Words with optional hyphens
    | \$?\d+(?:\.\d+)?%?        # Currency and percentages, e.g. $12.40, 82%
    | \.\.\.                    # Ellipsis
    | [\]\[.,;"'?():_\-`]       # Punctuation (escaped hyphen, backtick included)
)
"""

result = nltk.regexp_tokenize(text, pattern)
print(result)
