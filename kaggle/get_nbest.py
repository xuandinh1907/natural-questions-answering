
from set_up_data_structure import NbestPrediction , UNMAPPED

def clean_text(tok_text):
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text

def get_nbest(prelim_predictions, crops, example, n_best_size):
    seen, nbest = set(), []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        crop = crops[pred.crop_index]
        orig_doc_start, orig_doc_end = -1, -1
        # non-null
        if pred.start_index > 0:
            # Long answer has no end_index. We still generate some text to check
            if pred.end_index == -1:
                tok_tokens = crop.tokens[pred.start_index: pred.start_index + 11]
            else:
                tok_tokens = crop.tokens[pred.start_index: pred.end_index + 1]
            tok_text = " ".join(tok_tokens)
            tok_text = clean_text(tok_text)

            orig_doc_start = int(crop.token_to_orig_map[pred.start_index])
            if pred.end_index == -1:
                orig_doc_end = orig_doc_start + 10
            else:
                orig_doc_end = int(crop.token_to_orig_map[pred.end_index])

            final_text = tok_text
            if final_text in seen:
                continue

        else:
            final_text = ""

        seen.add(final_text)
        nbest.append(NbestPrediction(
            text=final_text,
            start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=pred.start_index, end_index=pred.end_index,
            orig_doc_start=orig_doc_start, orig_doc_end=orig_doc_end,
            crop_index=pred.crop_index))

    # Degenerate case. I never saw this happen.
    if len(nbest) in (0, 1):
        nbest.insert(0, NbestPrediction(text="empty",
            start_logit=0.0, end_logit=0.0,
            start_index=-1, end_index=-1,
            orig_doc_start=-1, orig_doc_end=-1,
            crop_index=UNMAPPED))

    assert len(nbest) >= 1
    return nbest