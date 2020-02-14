import argparse
import pickle
import os
import tensorflow as tf
from convert_nq_to_squad import convert_nq_to_squad
from read_nq_examples import read_nq_examples
from convert_examples_to_crops import convert_examples_to_crops

def get_convert_args():
    convert_args = argparse.Namespace()
    convert_args.fn = '../input/TensorFlow-2.0-Question-Answering/simplified-nq-test.jsonl'
    convert_args.version = 'v0.0.1'
    convert_args.prefix = 'nq'
    convert_args.num_samples = 1_000_000
    convert_args.val_ids = None
    convert_args.do_enumerate = False
    convert_args.do_not_dump = True
    convert_args.num_max_tokens = 400_000
    return convert_args

def load_and_cache_crops(args, tokenizer, evaluate=False):
    # Load data crops from cache or dataset file
    do_cache = False
    cached_crops_fn = 'cached_{}_{}.pkl'.format('test', str(args.max_seq_length))
    if os.path.exists(cached_crops_fn) and do_cache:
        print("Loading crops from cached file %s", cached_crops_fn)
        with open(cached_crops_fn, "rb") as f:
            crops = pickle.load(f)
    else:
        entries = convert_nq_to_squad(get_convert_args())
        examples_gen = read_nq_examples(entries, is_training=not evaluate)
        crops = convert_examples_to_crops(examples_gen=examples_gen,
                                          tokenizer=tokenizer,
                                          max_seq_length=args.max_seq_length,
                                          doc_stride=args.doc_stride,
                                          max_query_length=args.max_query_length,
                                          is_training=not evaluate,
                                          cls_token_segment_id=0,
                                          pad_token_segment_id=0,
                                          p_keep_impossible=args.p_keep_impossible if not evaluate else 1.0)
        if do_cache:
            with open(cached_crops_fn, "wb") as f:
                pickle.dump(crops, f)

    # stack
    all_input_ids = tf.stack([c.input_ids for c in crops], 0)
    all_attention_mask = tf.stack([c.attention_mask for c in crops], 0)
    all_token_type_ids = tf.stack([c.token_type_ids for c in crops], 0)

    if evaluate:
        dataset = [all_input_ids, all_attention_mask, all_token_type_ids]
    else:
        all_start_positions = tf.convert_to_tensor([f.start_position for f in crops], dtype=tf.int32)
        all_end_positions = tf.convert_to_tensor([f.end_position for f in crops], dtype=tf.int32)
        all_long_positions = tf.convert_to_tensor([f.long_position for f in crops], dtype=tf.int32)
        dataset = [all_input_ids, all_attention_mask, all_token_type_ids,
            all_start_positions, all_end_positions, all_long_positions]

    return dataset, crops, entries