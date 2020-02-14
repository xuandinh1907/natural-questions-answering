import argparse
import random
import os
import numpy as np
import tensorflow as tf
from get_add_tokens import get_add_tokens
from model import MODEL_CLASSES
from submit import submit

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_config",
        default="../input/transformers_cache/bert_large_uncased_config.json", type=str)
    parser.add_argument("--checkpoint_dir", default="../input/nq_bert_uncased_68", type=str)
    parser.add_argument("--vocab_txt", default="../input/transformers_cache/bert_large_uncased_vocab.txt", type=str)

    # Other parameters
    parser.add_argument('--short_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument('--long_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=256, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--per_tpu_eval_batch_size", default=4, type=int)
    parser.add_argument("--n_best_size", default=10, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p_keep_impossible', type=float,
                        default=0.1, help="The fraction of impossible"
                        " samples to keep.")
    parser.add_argument('--do_enumerate', action='store_true')

    args, _ = parser.parse_known_args()
    assert args.model_type not in ('xlnet', 'xlm'), f'Unsupported model_type: {args.model_type}'

    # Set seed
    set_seed(args)

    # Set cased / uncased
    config_basename = os.path.basename(args.model_config)
    if config_basename.startswith('bert'):
        do_lower_case = 'uncased' in config_basename
    elif config_basename.startswith('roberta'):
        # https://github.com/huggingface/transformers/pull/1386/files
        do_lower_case = False

    # Set XLA
    # https://github.com/kamalkraj/ALBERT-TF2.0/blob/8d0cc211361e81a648bf846d8ec84225273db0e4/run_classifer.py#L136
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({'pin_to_host_optimization': False})

    print("Training / evaluation parameters ", args)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_json_file(args.model_config)
    tokenizer = tokenizer_class(args.vocab_txt, do_lower_case=do_lower_case)
    tags = get_add_tokens(do_enumerate=args.do_enumerate)
    num_added = tokenizer.add_tokens(tags, offset=1)
    print(f"Added {num_added} tokens")
    print("Evaluate the following checkpoint: ", args.checkpoint_dir)
    weights_fn = os.path.join(args.checkpoint_dir, 'weights.h5')
    model = model_class(config)
    model(model.dummy_inputs, training=False)
    model.load_weights(weights_fn)

    # Evaluate
    result = submit(args, model, tokenizer)
    print("Result: {}".format(result))


main()