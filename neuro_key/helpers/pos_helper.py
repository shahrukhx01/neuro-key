from enum import Enum

# helpers for Embed Rank paper


class SupportedLanguages(str, Enum):
    """Supported languages for the NeuroKey application."""

    EN = "en"
    DE = "de"
    FR = "fr"


"""Language to POS grammar mapping."""
POS_GRAMMAR = {
    SupportedLanguages.EN: """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)""",
    SupportedLanguages.DE: """
NBAR:
        {<JJ|CARD>*<NN.*>+}  # [Adjective(s) or Article(s) or Posessive pronoun](optional) + Noun(s)
        {<NN>+<PPOSAT><JJ|CARD>*<NN.*>+}

NP:
{<NBAR><APPR|APPRART><ART>*<NBAR>}# Above, connected with APPR and APPART (beim vom)
{<NBAR>+}
""",
    SupportedLanguages.FR: """  NP:
        {<NN.*|JJ>*<NN.*>+<JJ>*}  # Adjective(s)(optional) + Noun(s) + Adjective(s)(optional)""",
}

POS_CONSIDERED_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ"}
