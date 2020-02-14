import logging
import collections
from collections import namedtuple

logger = logging.getLogger(__name__)


NQExample = collections.namedtuple("NQExample", [
    "qas_id", "question_text", "doc_tokens", "orig_answer_text",
    "start_position", "end_position", "long_position",
    "short_is_impossible", "long_is_impossible", "crop_start"])

Crop = collections.namedtuple("Crop", ["example_id","unique_id", "example_index", "doc_span_index",
    "tokens", "token_to_orig_map", "token_is_max_context",
    "input_ids", "attention_mask", "token_type_ids",
    "paragraph_len", "start_position", "end_position", "long_position",
    "short_is_impossible", "long_is_impossible"])

LongAnswerCandidate = collections.namedtuple('LongAnswerCandidate', [
    'start_token', 'end_token', 'top_level'])

DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

PrelimPrediction = collections.namedtuple("PrelimPrediction",
    ["crop_index", "start_index", "end_index", "start_logit", "end_logit"])

NbestPrediction = collections.namedtuple("NbestPrediction", [
    "text", "start_logit", "end_logit",
    "start_index", "end_index",
    "orig_doc_start", "orig_doc_end", "crop_index"])

RawResult = namedtuple("RawResult", ["unique_id", "start_logits", "end_logits",
    "long_logits"])

UNMAPPED = -123
CLS_INDEX = 0