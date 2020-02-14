import pandas as pd
from set_up_data_structure import logger

def convert_preds_to_df(preds, candidates):
  num_found_long, num_searched_long = 0, 0
  df = {'example_id': [], 'PredictionString': []}
  for example_id, pred in preds.items():
    short_text, start_token, end_token, long_text, long_token = pred
    df['example_id'].append(example_id + '_short')
    short_answer = ''
    if start_token != -1:
      # +1 is required to make the token inclusive
      short_answer = f'{start_token}:{end_token + 1}'
    df['PredictionString'].append(short_answer)

    # print(entry['document_text'].split(' ')[start_token: end_token + 1])
    # find the long answer
    long_answer = ''
    found_long = False
    min_dist = 1_000_000
    if long_token != -1:
      num_searched_long += 1
      for candidate in candidates[example_id]:
        cstart, cend = candidate.start_token, candidate.end_token
        dist = abs(cstart - long_token)
        if dist < min_dist:
          min_dist = dist
        if long_token == cstart:
          long_answer = f'{cstart}:{cend}'
          found_long = True
          break

      if found_long:
        num_found_long += 1
      else:
        logger.info(f"Not found: {min_dist}")

    df['example_id'].append(example_id + '_long')
    df['PredictionString'].append(long_answer)

  df = pd.DataFrame(df)
  print(f'Found {num_found_long} of {num_searched_long} (total {len(preds)})')
  return df