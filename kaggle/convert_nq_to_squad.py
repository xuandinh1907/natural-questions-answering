import numpy as np
import json
import argparse
import pandas as pd
from enumerate_tags import enumerate_tags


def convert_nq_to_squad(args=None):
  np.random.seed(123)
  if args is None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default='simplified-nq-train.jsonl')
    parser.add_argument('--version', type=str, default='v1.0.2')
    parser.add_argument('--prefix', type=str, default='nq')
    parser.add_argument('--p_val', type=float, default=0.1)
    parser.add_argument('--crop_len', type=int, default=2_500)
    parser.add_argument('--num_samples', type=int, default=1_000_000)
    parser.add_argument('--val_ids', type=str, default='val_ids.csv')
    parser.add_argument('--do_enumerate', action='store_true')
    parser.add_argument('--do_not_dump', action='store_true')
    parser.add_argument('--num_max_tokens', type=int, default=400_000)
    args = parser.parse_args()

  is_train = 'train' in args.fn
  if is_train:
    train_fn = f'{args.prefix}-train-{args.version}.json'
    val_fn = f'{args.prefix}-val-{args.version}.json'
    print(f'Converting {args.fn} to {train_fn} & {val_fn} ... ')
  else:
    test_fn = f'{args.prefix}-test-{args.version}.json'
    print(f'Converting {args.fn} to {test_fn} ... ')

  if args.val_ids:
    val_ids = set(str(x) for x in pd.read_csv(args.val_ids)['val_ids'].values)
  else:
    val_ids = set()

  entries = []
  smooth = 0.999
  total_split_len, long_split_len = 0., 0.
  long_end = 0.
  num_very_long, num_yes_no, num_short_dropped, num_trimmed = 0, 0, 0, 0
  num_short_possible, num_long_possible = 0, 0
  max_end_token = -1
  orig_data = {}
  with open(args.fn) as f:
    entry = {}
    for kk, line in enumerate(f):
      if kk >= args.num_samples:
        break

      data = json.loads(line)
      data_cpy = data.copy()
      example_id = str(data_cpy.pop('example_id'))
      data_cpy['document_text'] = ''
      orig_data[example_id] = data_cpy
      url = 'MISSING' if not is_train else data['document_url']
      # progress.write(f'############ {url} ###############')
      document_text = data['document_text']
      document_text_split = document_text.split(' ')
      # trim super long
      if len(document_text_split) > args.num_max_tokens:
        num_trimmed += 1
        document_text_split = document_text_split[:args.num_max_tokens]

      if args.do_enumerate:
        document_text_split = enumerate_tags(document_text_split)
      question = data['question_text']  # + '?'
      annotations = [None] if not is_train else data['annotations']
      assert len(annotations) == 1, annotations
      # User str keys!
      example_id = str(data['example_id'])
      candidates = data['long_answer_candidates']
      if not is_train:
        qa = {'question': question, 'id': example_id, 'crop_start': 0}
        context = ' '.join(document_text_split)

      else:
        long_answer = annotations[0]['long_answer']
        long_answer_len = long_answer['end_token'] - long_answer['start_token']
        total_split_len = smooth * total_split_len + (1. - smooth) * len(
            document_text_split)
        long_split_len = smooth * long_split_len + (1. - smooth) * \
            long_answer_len
        if long_answer['end_token'] > 0:
          long_end = smooth * long_end + (1. - smooth) * long_answer['end_token']

        if long_answer['end_token'] > max_end_token:
          max_end_token = long_answer['end_token']


        short_answers = annotations[0]['short_answers']
        yes_no_answer = annotations[0]['yes_no_answer']
        if yes_no_answer != 'NONE':
          # progress.write(f'Skipping yes-no: {yes_no_answer}')
          num_yes_no += 1
          continue

        # print(f'Q: {question}')
        # print(f'L: {long_answer_str}')
        long_is_impossible = long_answer['start_token'] == -1
        if long_is_impossible:
          long_answer_candidate = np.random.randint(len(candidates))
        else:
          long_answer_candidate = long_answer['candidate_index']

        long_start_token = candidates[long_answer_candidate]['start_token']
        long_end_token = candidates[long_answer_candidate]['end_token']
        # generate crop based on tokens. Note that validation samples should
        # not be cropped as this won't reflect test set performance.
        if args.crop_len > 0 and example_id not in val_ids:
          crop_start = long_start_token - np.random.randint(int(args.crop_len * 0.75))
          if crop_start <= 0:
            crop_start = 0
            crop_start_len = -1
          else:
            crop_start_len = len(' '.join(document_text_split[:crop_start]))

          crop_end = crop_start + args.crop_len
        else:
          crop_start = 0
          crop_start_len = -1
          crop_end = 10_000_000

        is_very_long = False
        if long_end_token > crop_end:
          num_very_long += 1
          is_very_long = True
          # progress.write(f'{num_very_long}: Skipping very long answer {long_end_token}, {crop_end}')
          # continue

        document_text_crop_split = document_text_split[crop_start: crop_end]
        context = ' '.join(document_text_crop_split)
        # create long answer
        long_answers_ = []
        if not long_is_impossible:
          long_answer_pre_split = document_text_split[:long_answer[
              'start_token']]
          long_answer_start = len(' '.join(long_answer_pre_split)) - \
              crop_start_len
          long_answer_split = document_text_split[long_answer['start_token']:
              long_answer['end_token']]
          long_answer_text = ' '.join(long_answer_split)
          if not is_very_long:
            assert context[long_answer_start: long_answer_start + len(
                long_answer_text)] == long_answer_text, long_answer_text
          long_answers_ = [{'text': long_answer_text,
              'answer_start': long_answer_start}]

        # create short answers
        short_is_impossible = len(short_answers) == 0
        short_answers_ = []
        if not short_is_impossible:
          for short_answer in short_answers:
            short_start_token = short_answer['start_token']
            short_end_token = short_answer['end_token']
            if short_start_token >= crop_start + args.crop_len:
              num_short_dropped += 1
              continue
            short_answers_pre_split = document_text_split[:short_start_token]
            short_answer_start = len(' '.join(short_answers_pre_split)) - \
                crop_start_len
            short_answer_split = document_text_split[short_start_token: short_end_token]
            short_answer_text = ' '.join(short_answer_split)
            assert short_answer_text != ''

            # this happens if we crop and parts of the short answer overflow
            short_from_context = context[short_answer_start: short_answer_start + len(short_answer_text)]
            if short_from_context != short_answer_text:
              print(f'short diff: {short_from_context} vs {short_answer_text}')
            short_answers_.append({'text': short_from_context,
                'answer_start': short_answer_start})

        if len(short_answers_) == 0:
          short_is_impossible = True

        if not short_is_impossible:
          num_short_possible += 1
        if not long_is_impossible:
          num_long_possible += 1

        qa = {'question': question,
            'short_answers': short_answers_, 'long_answers': long_answers_,
            'id': example_id, 'short_is_impossible': short_is_impossible,
            'long_is_impossible': long_is_impossible,
            'crop_start': crop_start}

      paragraph = {'qas': [qa], 'context': context}
      entry = {'title': url, 'paragraphs': [paragraph]}
      entries.append(entry)

  
  # shuffle to test remaining code
  np.random.shuffle(entries)

  if is_train:
    train_entries, val_entries = [], []
    for entry in entries:
      if entry['paragraphs'][0]['qas'][0]['id'] not in val_ids:
        train_entries.append(entry)
      else:
        val_entries.append(entry)

    for out_fn, entries in [(train_fn, train_entries), (val_fn, val_entries)]:
      if not args.do_not_dump:
        with open(out_fn, 'w') as f:
          json.dump({'version': args.version, 'data': entries}, f)

#       save val in competition csv format
      if 'val' in out_fn:
        val_example_ids, val_strs = [], []
        for entry in entries:
          example_id = entry['paragraphs'][0]['qas'][0]['id']
          short_answers = orig_data[example_id]['annotations'][0][
              'short_answers']
          sa_str = ''
          for si, sa in enumerate(short_answers):
            sa_str += f'{sa["start_token"]}:{sa["end_token"]}'
            if si < len(short_answers) - 1:
              sa_str += ' '
          val_example_ids.append(example_id + '_short')
          val_strs.append(sa_str)

          la = orig_data[example_id]['annotations'][0][
              'long_answer']
          la_str = ''
          if la['start_token'] > 0:
            la_str += f'{la["start_token"]}:{la["end_token"]}'
          val_example_ids.append(example_id + '_long')
          val_strs.append(la_str)

        val_df = pd.DataFrame({'example_id': val_example_ids,
            'PredictionString': val_strs})
        val_csv_fn = f'{args.prefix}-val-{args.version}.csv'
        val_df.to_csv(val_csv_fn, index=False, columns=['example_id',
            'PredictionString'])
        print(f'Wrote csv to {val_csv_fn}')

  else:
    if not args.do_not_dump:
      with open(test_fn, 'w') as f:
        json.dump({'version': args.version, 'data': entries}, f)

  if args.val_ids:
    print(f'Using val ids from: {args.val_ids}')
  return entries