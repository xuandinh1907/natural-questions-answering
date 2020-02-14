import math
import time
import gc
import tensorflow as tf 
from load_and_cache_crops import load_and_cache_crops
from set_up_data_structure import RawResult
from read_nq_examples import read_nq_examples
from write_predictions import write_predictions
from read_candidates import read_candidates
from convert_preds_to_df import convert_preds_to_df

def submit(args, model, tokenizer):
    csv_fn = 'submission.csv'
    # all_input_ids, all_attention_mask, all_token_type_ids, all_p_mask
    eval_dataset, crops, entries  = load_and_cache_crops(args, tokenizer, evaluate=True)
    args.eval_batch_size = args.per_tpu_eval_batch_size

    # pad dataset to multiple of `args.eval_batch_size`
    eval_dataset_length = len(eval_dataset[0])
    padded_length = math.ceil(eval_dataset_length / args.eval_batch_size) * args.eval_batch_size
    # num_pad = padded_length - eval_dataset[0].shape[0]
    for ti, t in enumerate(eval_dataset):
        pad_tensor = tf.expand_dims(tf.zeros_like(t[0]), 0)
        # pad_tensor = tf.repeat(pad_tensor, num_pad, 0)
        eval_dataset[ti] = tf.concat([t, pad_tensor], 0)

    # create eval dataset
    eval_ds = tf.data.Dataset.from_tensor_slices({
        'input_ids': tf.constant(eval_dataset[0]),
        'attention_mask': tf.constant(eval_dataset[1]),
        'token_type_ids': tf.constant(eval_dataset[2]),
        'example_index': tf.range(padded_length, dtype=tf.int32)

    })
    eval_ds = eval_ds.batch(batch_size=args.eval_batch_size, drop_remainder=True)
    # eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)
    # eval_ds = strategy.experimental_distribute_dataset(eval_ds)

    # eval
    print("***** Running evaluation *****")
    print("  Num examples =  ", eval_dataset_length)
    print("  Batch size =  ", args.eval_batch_size)

    @tf.function
    def predict_step(batch):
        outputs = model(batch, training=False)
        return outputs

    all_results = []
    tic = time.time()
    for batch_ind, batch in enumerate(eval_ds):
        example_indexes = batch['example_index']
        outputs = predict_step(batch)
        batched_start_logits = outputs[0].numpy()
        batched_end_logits = outputs[1].numpy()
        batched_long_logits = outputs[2].numpy()
        for i, example_index in enumerate(example_indexes):
            # filter out padded samples
            if example_index >= eval_dataset_length:
                continue

            eval_crop = crops[example_index]
            unique_id = int(eval_crop.unique_id)
            start_logits = batched_start_logits[i].tolist()
            end_logits = batched_end_logits[i].tolist()
            long_logits = batched_long_logits[i].tolist()

            result = RawResult(unique_id=unique_id,
                               start_logits=start_logits,
                               end_logits=end_logits,
                               long_logits=long_logits)
            all_results.append(result)

    eval_time = time.time() - tic
    print("  Evaluation done in total %f secs (%f sec per example)",
        eval_time, eval_time / padded_length)
    examples_gen = read_nq_examples(entries, is_training=False)
    print("***** Writing predictions *****")
    preds = write_predictions(examples_gen, crops, all_results, args.n_best_size,
                              args.max_answer_length,
                              None, None, None,
                              args.verbose_logging,
                              args.short_null_score_diff_threshold, args.long_null_score_diff_threshold)
    del crops, all_results
    gc.collect()
    print("***** Starting reading candidates *****")
    candidates = read_candidates(['../input/TensorFlow-2.0-Question-Answering/simplified-nq-test.jsonl'], do_cache=False)
    sub = convert_preds_to_df(preds, candidates).sort_values('example_id')
    sub.to_csv(csv_fn, index=False, columns=['example_id', 'PredictionString'])
    print(f'***** Wrote submission to {csv_fn} *****')
    result = {}
    return result