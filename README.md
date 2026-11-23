# English–Hindi Machine Translation

My work for the *Language as Data* course (MA Text Mining, Vrije Universiteit Amsterdam), centred on one question: why is English→Hindi machine translation hard, and how can data help? My final paper is in [Main-Papers](https://github.com/Mattzoki/Main-Papers), under the title *Lost in Translation? Making English and Hindi Work Together*.

## What's here

### Linguistic phenomena (Assignment 1)
Five English→Hindi challenges, each with how human translators handle it and how to probe a model for it:
- definite and indefinite articles (Hindi has none)
- phrasal verbs
- passive voice
- idioms
- grammatical gender

### Dataset (Assignment 2)
- A parallel English–Hindi corpus built from Bible translations (`English.xml`, `Hindi.xml`), aligned by sentence ID with custom code into pairs and spot-checked by hand.
- `dataset_en_hi.xlsx`: English–Hindi sentence data in a spreadsheet.
- Data statement and annotation of the phenomena above.

### LLM data augmentation and evaluation
- `llamma3-machine-translation.ipynb`: runs a quantised Llama 3.1 model locally (GGUF via `llama-cpp-python`, accessed through an OpenAI-style client) to generate English sentences with Hindi translations, for example sentences that contain idioms.
- Introduces BLEU with NLTK and custom n-gram weights, to judge how well idioms and word order survive translation.
- The final paper looks at word order (SVO vs SOV), Hindi morphological richness and idioms, and evaluates a pretrained model zero-shot with CTranslate2 and BLEU.

### Course labs
`LaD/Labs/` contains the lab notebooks (crawling, linguistic units, sentence analysis, terminology, language modelling) and a JoeyNMT translation-training tutorial.

## Layout

| Path | Contents |
| --- | --- |
| `llamma3-machine-translation.ipynb` | LLM data augmentation and BLEU notebook |
| `LaD/Labs/` | Lab notebooks and assignment data (`Assignments/`) |
| `submissions/` | Written assignments and lab write-ups |
| `news-commentary-v18.en-hi.tsv.gz`, `wmt14.en-hi.en`, `wmt_devsets.zip` | Public English–Hindi data and WMT dev sets |
| `*.pdf` | Papers I used as references |

## Setup

```bash
pip install nltk openai jupyter pandas
# for the local Llama server used in the notebook:
pip install llama-cpp-python "fastapi[all]" sse_starlette starlette_context pydantic_settings
python -m llama_cpp.server --model ./models/<your-gguf-model>.gguf --n_ctx 8192
```

You need to download a GGUF model yourself (for example Llama 3.1 8B Instruct, quantised). The notebook expects the server at `http://localhost:8000/v1`.

## Context

Hindi is a morphologically rich, SOV language without articles, so translation from English needs restructuring, not word-for-word mapping. Idioms and gender show this most clearly.
