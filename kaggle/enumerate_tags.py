def enumerate_tags(text_split):
  """Reproduce the preprocessing from:
  A BERT Baseline for the Natural Questions (https://arxiv.org/pdf/1901.08634.pdf)

  We introduce special markup tokens in the doc-ument  to  give  the  model
  a  notion  of  which  part of the document it is reading.  The special
  tokens we introduced are of the form “[Paragraph=N]”,“[Table=N]”, and “[List=N]”
  at the beginning ofthe N-th paragraph,  list and table respectively
  in the document. This decision was based on the observation that the first
  few paragraphs and tables in the document are much more likely than the rest
  of the document to contain the annotated answer and so the model could benefit
  from knowing whether it is processing one of these passages.

  We deviate as follows: Tokens are only created for the first 10 times. All other
  tokens are the same. We only add `special_tokens`. These two are added as they
  make 72.9% + 19.0% = 91.9% of long answers.
  (https://github.com/google-research-datasets/natural-questions)
  """
  special_tokens = ['<P>', '<Table>']
  special_token_counts = [0 for _ in range(len(special_tokens))]
  for index, token in enumerate(text_split):
    for special_token_index, special_token in enumerate(special_tokens):
      if token == special_token:
        cnt = special_token_counts[special_token_index]
        if cnt <= 10:
          text_split[index] = f'<{special_token[1: -1]}{cnt}>'
        special_token_counts[special_token_index] = cnt + 1

  return text_split