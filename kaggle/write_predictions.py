import collections
import json
import numpy as np
from set_up_data_structure import CLS_INDEX , logger , UNMAPPED , PrelimPrediction
from get_nbest import clean_text , get_nbest

def write_predictions(examples_gen, all_crops, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      short_null_score_diff, long_null_score_diff):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    # create indexes
    example_index_to_crops = collections.defaultdict(list)
    for crop in all_crops:
        example_index_to_crops[crop.example_index].append(crop)
    unique_id_to_result = {result.unique_id: result for result in all_results}

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    short_num_empty, long_num_empty = 0, 0
    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            logger.info(f'[{example_index}]: {short_num_empty} short and {long_num_empty} long empty')

        crops = example_index_to_crops[example_index]
        short_prelim_predictions, long_prelim_predictions = [], []
        for crop_index, crop in enumerate(crops):
            assert crop.unique_id in unique_id_to_result, f"{crop.unique_id}"
            result = unique_id_to_result[crop.unique_id]
            # get the `n_best_size` largest indexes
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array#23734295
            start_indexes = np.argpartition(result.start_logits, -n_best_size)[-n_best_size:]
            start_indexes = [int(x) for x in start_indexes]
            end_indexes = np.argpartition(result.end_logits, -n_best_size)[-n_best_size:]
            end_indexes = [int(x) for x in end_indexes]

            # create short answers
            for start_index in start_indexes:
                if start_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[start_index] == UNMAPPED:
                    continue
                if not crop.token_is_max_context[start_index]:
                    continue

                for end_index in end_indexes:
                    if end_index >= len(crop.tokens):
                        continue
                    if crop.token_to_orig_map[end_index] == UNMAPPED:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    short_prelim_predictions.append(PrelimPrediction(
                        crop_index=crop_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

            long_indexes = np.argpartition(result.long_logits, -n_best_size)[-n_best_size:].tolist()
            for long_index in long_indexes:
                if long_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[long_index] == UNMAPPED:
                    continue
                # TODO(see--): Is this needed?
                # -> Yep helps both short and long by about 0.1
                if not crop.token_is_max_context[long_index]:
                    continue
                long_prelim_predictions.append(PrelimPrediction(
                    crop_index=crop_index,
                    start_index=long_index, end_index=-1,
                    start_logit=result.long_logits[long_index],
                    end_logit=result.long_logits[long_index]))

        short_prelim_predictions = sorted(short_prelim_predictions,
            key=lambda x: x.start_logit + x.end_logit, reverse=True)

        short_nbest = get_nbest(short_prelim_predictions, crops,
            example, n_best_size)

        short_best_non_null = None
        for entry in short_nbest:
            if short_best_non_null is None:
                if entry.text != "":
                    short_best_non_null = entry

        long_prelim_predictions = sorted(long_prelim_predictions,
            key=lambda x: x.start_logit, reverse=True)

        long_nbest = get_nbest(long_prelim_predictions, crops,
            example, n_best_size)

        long_best_non_null = None
        for entry in long_nbest:
            if long_best_non_null is None:
                if entry.text != "":
                    long_best_non_null = entry

        nbest_json = {'short': [], 'long': []}
        for kk, entries in [('short', short_nbest), ('long', long_nbest)]:
            for i, entry in enumerate(entries):
                output = {}
                output["text"] = entry.text
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["start_index"] = entry.start_index
                output["end_index"] = entry.end_index
                output["orig_doc_start"] = entry.orig_doc_start
                output["orig_doc_end"] = entry.orig_doc_end
                nbest_json[kk].append(output)

        assert len(nbest_json['short']) >= 1
        assert len(nbest_json['long']) >= 1

        # We use the [CLS] score of the crop that has the maximum positive score
        # long_score_diff = min_long_score_null - long_best_non_null.start_logit
        # Predict "" if null score - the score of best non-null > threshold
        try:
            crop_unique_id = crops[short_best_non_null.crop_index].unique_id
            start_score_null = unique_id_to_result[crop_unique_id].start_logits[CLS_INDEX]
            end_score_null = unique_id_to_result[crop_unique_id].end_logits[CLS_INDEX]
            short_score_null = start_score_null + end_score_null
            short_score_diff = short_score_null - (short_best_non_null.start_logit +
                short_best_non_null.end_logit)

            if short_score_diff > short_null_score_diff:
                final_pred = ("", -1, -1)
                short_num_empty += 1
            else:
                final_pred = (short_best_non_null.text, short_best_non_null.orig_doc_start,
                    short_best_non_null.orig_doc_end)
        except Exception as e:
            print(e)
            final_pred = ("", -1, -1)
            short_num_empty += 1

        try:
            long_score_null = unique_id_to_result[crops[
                long_best_non_null.crop_index].unique_id].long_logits[CLS_INDEX]
            long_score_diff = long_score_null - long_best_non_null.start_logit
            scores_diff_json[example.qas_id] = {'short_score_diff': short_score_diff,
                'long_score_diff': long_score_diff}

            if long_score_diff > long_null_score_diff:
                final_pred += ("", -1)
                long_num_empty += 1
                # print(f"LONG EMPTY: {round(long_score_null, 2)} vs "
                #     f"{round(long_best_non_null.start_logit, 2)} (th {long_null_score_diff})")

            else:
                final_pred += (long_best_non_null.text, long_best_non_null.orig_doc_start)

        except Exception as e:
            print(e)
            final_pred += ("", -1)
            long_num_empty += 1

        all_predictions[example.qas_id] = final_pred
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=2))

    if output_nbest_file is not None:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=2))

    if output_null_log_odds_file is not None:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=2))

    logger.info(f'{short_num_empty} short and {long_num_empty} long empty of'
        f' {example_index}')
    return all_predictions