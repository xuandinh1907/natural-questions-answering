import os
import json
import pickle
from set_up_data_structure import LongAnswerCandidate

def read_candidates(candidate_files, do_cache=True):
    assert isinstance(candidate_files, (tuple, list)), candidate_files
    for fn in candidate_files:
        assert os.path.exists(fn), f'Missing file {fn}'
    cache_fn = 'candidates.pkl'

    candidates = {}
    if not os.path.exists(cache_fn):
        for fn in candidate_files:
            with open(fn) as f:
                for line in f:
                    entry = json.loads(line)
                    example_id = str(entry['example_id'])
                    cnds = entry.pop('long_answer_candidates')
                    cnds = [LongAnswerCandidate(c['start_token'], c['end_token'],
                            c['top_level']) for c in cnds]
                    candidates[example_id] = cnds

        if do_cache:
            with open(cache_fn, 'wb') as f:
                pickle.dump(candidates, f)
    else:
        print(f'Loading from cache: {cache_fn}')
        with open(cache_fn, 'rb') as f:
            candidates = pickle.load(f)

    return candidates